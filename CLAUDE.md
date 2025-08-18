# Claude Code Instructions for Inference Systems Lab

## MANDATORY BUILD TESTING REQUIREMENT

**🚨 CRITICAL: YOU MUST TEST BUILD AFTER EVERY CODE CHANGE 🚨**

### Required Build Test Sequence:
1. After making ANY changes to source files (.cpp, .hpp, .h), you MUST run:
   ```bash
   cd /Users/dbjones/ng/dbjwhs/inference-systems-lab/build && make -j4
   ```

2. If build fails:
   - STOP immediately
   - Fix the compilation errors
   - Test build again
   - Only proceed when build succeeds

3. Never commit code that doesn't build successfully

### Why This Matters:
- Static analysis fixes often break compilation due to naming changes
- Other files may reference old function/variable names
- Build failures block user's development workflow
- Manual fixes require understanding cross-file dependencies

### Enforcement:
- This file (CLAUDE.md) serves as a persistent reminder
- Check this file at the start of each session
- Reference these requirements when making code changes
- User will reject any work that breaks the build

### Build Commands:
- **Full build**: `cd build && make -j4`
- **Clean build**: `cd build && make clean && make -j4`
- **Single target**: `cd build && make <target>`

### Current Project Status:
- Project uses CMake build system
- Build directory: `/Users/dbjones/ng/dbjwhs/inference-systems-lab/build`
- Main issues: Function/method naming changes break cross-file references
- Pre-commit hooks check formatting and static analysis on committed files only

## Static Analysis Fix Strategy

### Systematic Approach:
1. **Always build test first** - before any changes
2. Fix one file at a time
3. Build test after each file
4. Fix any compilation errors immediately
5. Only move to next file when current file builds successfully

### Files Modified So Far:
- ✅ `logging.hpp` - completed, builds successfully
- ✅ `inference_types.hpp` - completed, but may have broken other files
- 🚨 Need to fix compilation issues caused by inference_types changes

### Next Steps:
1. Test current build status
2. Fix any compilation errors from recent changes
3. Resume systematic static analysis improvements
4. Always test build between changes

## User Requirements Summary:
- Build must work after every change
- Test with `cd build && make` before proceeding
- Fix compilation errors immediately
- No commits that break the build
- Systematic, careful approach over speed
