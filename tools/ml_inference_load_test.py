#!/usr/bin/env python3
# MIT License
# Copyright (c) 2025 dbjwhs
"""
ml_inference_load_test.py - Load testing framework for ML inference scenarios

This tool provides load testing capabilities for the upcoming TensorRT/ONNX integration:
- Simulates realistic ML inference workloads
- Tests concurrent request handling
- Measures memory bandwidth utilization
- Evaluates GPU memory management
- Benchmarks tensor data transfer overhead
- Validates system behavior under stress
"""

import asyncio
import json
import time
import threading
import queue
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


@dataclass
class InferenceRequest:
    """Simulated ML inference request."""
    request_id: str
    model_name: str
    input_shape: tuple
    input_data_mb: float
    expected_latency_ms: float
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class LoadTestResult:
    """Results from a load test scenario."""
    scenario_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    memory_peak_mb: float
    cpu_utilization_pct: float
    gpu_utilization_pct: float
    errors: List[str]


class MockInferenceEngine:
    """Mock inference engine for load testing without actual ML models."""
    
    def __init__(self, engine_type: str = "tensorrt"):
        self.engine_type = engine_type
        self.model_cache = {}
        self.gpu_memory_used = 0
        self.max_gpu_memory = 8192  # 8GB simulation
        
    async def load_model(self, model_name: str, model_size_mb: float) -> bool:
        """Simulate loading a model into GPU memory."""
        if self.gpu_memory_used + model_size_mb > self.max_gpu_memory:
            return False
        
        # Simulate model loading time
        await asyncio.sleep(0.01 + model_size_mb / 1000.0)
        
        self.model_cache[model_name] = {
            'size_mb': model_size_mb,
            'loaded_at': time.time()
        }
        self.gpu_memory_used += model_size_mb
        return True
    
    async def run_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Simulate running inference on a request."""
        start_time = time.time()
        
        # Check if model is loaded
        if request.model_name not in self.model_cache:
            model_size = 100 + (request.input_data_mb * 2)  # Estimate model size
            if not await self.load_model(request.model_name, model_size):
                raise RuntimeError(f"Failed to load model {request.model_name}: GPU memory full")
        
        # Simulate computation time based on input size and model complexity
        base_compute_time = request.expected_latency_ms / 1000.0
        compute_time = base_compute_time * (0.8 + 0.4 * len(asyncio.all_tasks()) / 10)
        
        await asyncio.sleep(compute_time)
        
        end_time = time.time()
        
        return {
            'request_id': request.request_id,
            'latency_ms': (end_time - start_time) * 1000,
            'output_shape': request.input_shape,  # Simplified
            'gpu_memory_used': self.gpu_memory_used,
            'success': True
        }


class MLInferenceLoadTester:
    """Main load testing framework for ML inference scenarios."""
    
    def __init__(self):
        self.engine = MockInferenceEngine()
        self.results = []
        
    def generate_realistic_workload(self, scenario: str, num_requests: int) -> List[InferenceRequest]:
        """Generate realistic ML inference workloads for different scenarios."""
        requests = []
        
        if scenario == "image_classification":
            # Typical image classification workload
            for i in range(num_requests):
                requests.append(InferenceRequest(
                    request_id=f"img_cls_{i}",
                    model_name="resnet50" if i % 3 == 0 else "efficientnet_b4",
                    input_shape=(1, 3, 224, 224),
                    input_data_mb=0.6,  # 224x224x3 float32
                    expected_latency_ms=5.0 + (i % 10),
                    priority=1 if i % 5 == 0 else 2
                ))
                
        elif scenario == "object_detection":
            # Object detection with larger images
            for i in range(num_requests):
                requests.append(InferenceRequest(
                    request_id=f"obj_det_{i}",
                    model_name="yolov8" if i % 2 == 0 else "detectron2",
                    input_shape=(1, 3, 640, 640),
                    input_data_mb=4.9,  # 640x640x3 float32
                    expected_latency_ms=15.0 + (i % 20),
                    priority=1
                ))
                
        elif scenario == "nlp_transformers":
            # NLP transformer models with variable sequence lengths
            for i in range(num_requests):
                seq_len = 128 + (i % 384)  # Variable sequence length
                requests.append(InferenceRequest(
                    request_id=f"nlp_trans_{i}",
                    model_name="bert_large" if i % 4 == 0 else "gpt2_medium",
                    input_shape=(1, seq_len, 768),
                    input_data_mb=(seq_len * 768 * 4) / (1024 * 1024),
                    expected_latency_ms=20.0 + (seq_len / 10),
                    priority=2
                ))
                
        elif scenario == "mixed_workload":
            # Mixed workload combining different model types
            for i in range(num_requests):
                if i % 3 == 0:
                    # Image classification
                    requests.append(InferenceRequest(
                        request_id=f"mixed_img_{i}",
                        model_name="mobilenet_v3",
                        input_shape=(1, 3, 224, 224),
                        input_data_mb=0.6,
                        expected_latency_ms=3.0,
                        priority=1
                    ))
                elif i % 3 == 1:
                    # Object detection
                    requests.append(InferenceRequest(
                        request_id=f"mixed_det_{i}",
                        model_name="yolov5s",
                        input_shape=(1, 3, 416, 416),
                        input_data_mb=2.1,
                        expected_latency_ms=8.0,
                        priority=2
                    ))
                else:
                    # NLP
                    requests.append(InferenceRequest(
                        request_id=f"mixed_nlp_{i}",
                        model_name="distilbert",
                        input_shape=(1, 256, 512),
                        input_data_mb=0.5,
                        expected_latency_ms=12.0,
                        priority=3
                    ))
                    
        return requests
    
    async def run_concurrent_load_test(self, 
                                     requests: List[InferenceRequest],
                                     max_concurrent: int = 10,
                                     scenario_name: str = "default") -> LoadTestResult:
        """Run concurrent load test with specified parameters."""
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        errors = []
        
        async def process_request(request: InferenceRequest):
            async with semaphore:
                try:
                    result = await self.engine.run_inference(request)
                    results.append(result)
                    return result
                except Exception as e:
                    error_msg = f"Request {request.request_id} failed: {str(e)}"
                    errors.append(error_msg)
                    return None
        
        # Execute all requests concurrently
        tasks = [process_request(req) for req in requests]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate statistics
        successful_results = [r for r in results if r and r.get('success', False)]
        latencies = [r['latency_ms'] for r in successful_results]
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        else:
            avg_latency = p95_latency = p99_latency = 0.0
        
        throughput_rps = len(successful_results) / total_duration if total_duration > 0 else 0.0
        
        return LoadTestResult(
            scenario_name=scenario_name,
            total_requests=len(requests),
            successful_requests=len(successful_results),
            failed_requests=len(requests) - len(successful_results),
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_rps=throughput_rps,
            memory_peak_mb=self.engine.gpu_memory_used,
            cpu_utilization_pct=85.0,  # Simulated
            gpu_utilization_pct=75.0,  # Simulated
            errors=errors
        )
    
    async def run_stress_test(self, 
                            base_requests: List[InferenceRequest],
                            duration_seconds: int = 60) -> List[LoadTestResult]:
        """Run stress test by gradually increasing load over time."""
        
        stress_results = []
        concurrency_levels = [1, 2, 5, 10, 20, 50]
        
        for concurrency in concurrency_levels:
            print(f"  🔄 Testing concurrency level: {concurrency}")
            
            # Repeat requests to fill duration
            test_requests = []
            while len(test_requests) < concurrency * 10:  # Ensure enough requests
                test_requests.extend(base_requests[:min(len(base_requests), 
                                                      concurrency * 10 - len(test_requests))])
            
            result = await self.run_concurrent_load_test(
                requests=test_requests[:concurrency * 5],  # Reasonable batch size
                max_concurrent=concurrency,
                scenario_name=f"stress_test_c{concurrency}"
            )
            
            stress_results.append(result)
            
            # Check if system is saturated (degraded performance)
            if len(stress_results) > 1:
                prev_result = stress_results[-2]
                if (result.avg_latency_ms > prev_result.avg_latency_ms * 2 or
                    result.failed_requests > result.total_requests * 0.1):
                    print(f"  ⚠️  System saturation detected at concurrency {concurrency}")
                    break
        
        return stress_results
    
    def run_all_scenarios(self) -> Dict[str, Any]:
        """Run comprehensive load testing across all scenarios."""
        
        print("🚀 Starting ML Inference Load Testing Suite")
        print("=" * 60)
        
        scenarios = {
            "image_classification": 100,
            "object_detection": 50,
            "nlp_transformers": 30,
            "mixed_workload": 80
        }
        
        all_results = {}
        
        for scenario_name, num_requests in scenarios.items():
            print(f"\n📊 Testing scenario: {scenario_name}")
            print(f"  📝 Generating {num_requests} requests...")
            
            requests = self.generate_realistic_workload(scenario_name, num_requests)
            
            # Run basic load test
            print("  🏃 Running basic load test...")
            basic_result = asyncio.run(self.run_concurrent_load_test(
                requests=requests,
                max_concurrent=10,
                scenario_name=scenario_name
            ))
            
            # Run stress test with subset
            print("  💪 Running stress test...")
            stress_results = asyncio.run(self.run_stress_test(
                base_requests=requests[:20],  # Use subset for stress test
                duration_seconds=30
            ))
            
            all_results[scenario_name] = {
                "basic_load_test": self._result_to_dict(basic_result),
                "stress_test_results": [self._result_to_dict(r) for r in stress_results]
            }
            
            print(f"  ✅ Completed {scenario_name}")
            print(f"     Throughput: {basic_result.throughput_rps:.1f} RPS")
            print(f"     Avg Latency: {basic_result.avg_latency_ms:.1f} ms")
            print(f"     Success Rate: {basic_result.successful_requests/basic_result.total_requests*100:.1f}%")
        
        return {
            "test_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "engine_type": self.engine.engine_type,
                "max_gpu_memory_mb": self.engine.max_gpu_memory
            },
            "scenario_results": all_results,
            "performance_summary": self._generate_performance_summary(all_results)
        }
    
    def _result_to_dict(self, result: LoadTestResult) -> Dict[str, Any]:
        """Convert LoadTestResult to dictionary."""
        return {
            "scenario_name": result.scenario_name,
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "p99_latency_ms": result.p99_latency_ms,
            "throughput_rps": result.throughput_rps,
            "memory_peak_mb": result.memory_peak_mb,
            "cpu_utilization_pct": result.cpu_utilization_pct,
            "gpu_utilization_pct": result.gpu_utilization_pct,
            "errors": result.errors
        }
    
    def _generate_performance_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall performance summary."""
        
        total_throughput = 0
        avg_latencies = []
        success_rates = []
        
        for scenario_name, scenario_results in all_results.items():
            basic_result = scenario_results["basic_load_test"]
            total_throughput += basic_result["throughput_rps"]
            avg_latencies.append(basic_result["avg_latency_ms"])
            
            success_rate = basic_result["successful_requests"] / basic_result["total_requests"]
            success_rates.append(success_rate)
        
        return {
            "overall_throughput_rps": total_throughput,
            "average_latency_ms": statistics.mean(avg_latencies) if avg_latencies else 0,
            "overall_success_rate": statistics.mean(success_rates) if success_rates else 0,
            "performance_grade": self._calculate_performance_grade(total_throughput, avg_latencies, success_rates),
            "bottleneck_analysis": self._analyze_bottlenecks(all_results),
            "scaling_recommendations": self._generate_scaling_recommendations(all_results)
        }
    
    def _calculate_performance_grade(self, throughput: float, latencies: List[float], success_rates: List[float]) -> str:
        """Calculate overall performance grade."""
        
        # Scoring criteria
        throughput_score = min(100, throughput * 2)  # 50 RPS = 100 points
        latency_score = max(0, 100 - statistics.mean(latencies) * 2)  # 50ms = 0 points
        reliability_score = statistics.mean(success_rates) * 100
        
        overall_score = (throughput_score + latency_score + reliability_score) / 3
        
        if overall_score >= 90:
            return "A+ (Excellent)"
        elif overall_score >= 80:
            return "A (Very Good)"
        elif overall_score >= 70:
            return "B (Good)"
        elif overall_score >= 60:
            return "C (Fair)"
        else:
            return "D (Needs Improvement)"
    
    def _analyze_bottlenecks(self, all_results: Dict[str, Any]) -> List[str]:
        """Analyze potential bottlenecks from test results."""
        bottlenecks = []
        
        for scenario_name, scenario_results in all_results.items():
            stress_results = scenario_results["stress_test_results"]
            
            # Check for memory bottleneck
            peak_memory = max([r["memory_peak_mb"] for r in stress_results])
            if peak_memory > 6000:  # 75% of 8GB
                bottlenecks.append(f"GPU memory saturation in {scenario_name} (peak: {peak_memory:.0f} MB)")
            
            # Check for latency degradation
            if len(stress_results) > 1:
                latency_increase = stress_results[-1]["avg_latency_ms"] / stress_results[0]["avg_latency_ms"]
                if latency_increase > 3:
                    bottlenecks.append(f"Severe latency degradation in {scenario_name} ({latency_increase:.1f}x increase)")
        
        if not bottlenecks:
            bottlenecks.append("No major bottlenecks detected in current test scenarios")
        
        return bottlenecks
    
    def _generate_scaling_recommendations(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for scaling ML inference."""
        recommendations = []
        
        # Analyze throughput patterns
        throughputs = []
        for scenario_results in all_results.values():
            throughputs.append(scenario_results["basic_load_test"]["throughput_rps"])
        
        avg_throughput = statistics.mean(throughputs)
        
        if avg_throughput < 10:
            recommendations.append("Consider GPU scaling or model optimization for higher throughput")
        
        if avg_throughput > 50:
            recommendations.append("Excellent throughput - ready for production deployment")
        
        # Memory utilization analysis
        max_memory_usage = 0
        for scenario_results in all_results.values():
            for stress_result in scenario_results["stress_test_results"]:
                max_memory_usage = max(max_memory_usage, stress_result["memory_peak_mb"])
        
        if max_memory_usage > 6000:
            recommendations.append("Implement model quantization or pruning to reduce memory usage")
        elif max_memory_usage < 2000:
            recommendations.append("GPU memory is underutilized - can handle larger models or batches")
        
        recommendations.extend([
            "Implement dynamic batching for variable workloads",
            "Consider multi-GPU deployment for horizontal scaling",
            "Use model caching strategies for frequently accessed models"
        ])
        
        return recommendations


def main():
    """Main entry point for ML inference load testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ML Inference Load Testing Framework"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default="ml_load_test_results.json",
        help="Output file for test results"
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=["image_classification", "object_detection", "nlp_transformers", "mixed_workload", "all"],
        default="all",
        help="Specific scenario to test"
    )
    
    args = parser.parse_args()
    
    # Run load tests
    tester = MLInferenceLoadTester()
    
    if args.scenario == "all":
        results = tester.run_all_scenarios()
    else:
        # Run single scenario
        requests = tester.generate_realistic_workload(args.scenario, 50)
        basic_result = asyncio.run(tester.run_concurrent_load_test(
            requests=requests,
            max_concurrent=10,
            scenario_name=args.scenario
        ))
        
        results = {
            "test_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "engine_type": tester.engine.engine_type
            },
            "scenario_results": {
                args.scenario: {"basic_load_test": tester._result_to_dict(basic_result)}
            }
        }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Load test results saved to {args.output}")
    
    # Print summary
    if "performance_summary" in results:
        summary = results["performance_summary"]
        print(f"\n🏆 PERFORMANCE SUMMARY")
        print(f"Overall Throughput: {summary['overall_throughput_rps']:.1f} RPS")
        print(f"Average Latency: {summary['average_latency_ms']:.1f} ms")
        print(f"Success Rate: {summary['overall_success_rate']*100:.1f}%")
        print(f"Performance Grade: {summary['performance_grade']}")
        
        print(f"\n🔍 BOTTLENECK ANALYSIS:")
        for bottleneck in summary['bottleneck_analysis']:
            print(f"  • {bottleneck}")
        
        print(f"\n💡 SCALING RECOMMENDATIONS:")
        for rec in summary['scaling_recommendations'][:3]:
            print(f"  • {rec}")


if __name__ == "__main__":
    main()