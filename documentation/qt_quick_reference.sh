#!/bin/bash
# Quick reference for running tests with Qt windows

echo "=== Qt Window Integration - Quick Reference ==="
echo ""

echo "1. Install PySide6 (if not already installed):"
echo "   conda activate vigil"
echo "   pip install PySide6"
echo ""

echo "2. Test Qt windows directly:"
echo "   python tests/test_qt_window.py"
echo ""

echo "3. Verify Qt integration:"
echo "   python tests/verify_qt_integration.py"
echo ""

echo "4. Run guidance tests (now using Qt automatically):"
echo "   python -m tests.test_guidance frames /path/to/frames"
echo ""

echo "5. Run guidance UI (now using Qt automatically):"
echo "   python src/guidance_ui.py --source simulator --frames-folder /path/to/frames"
echo ""

echo "Note: No code changes needed - Qt windows are now the default!"
echo "      All existing scripts automatically use QtImageWindow."
