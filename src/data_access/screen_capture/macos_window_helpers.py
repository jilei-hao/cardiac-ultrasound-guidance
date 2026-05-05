import sys

def is_macos():
  return sys.platform == "darwin"

if is_macos():
  from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly
  from AppKit import NSWorkspace

def get_window_bbox(window_name):
  """Find window position by name on macOS"""
  options = kCGWindowListOptionOnScreenOnly
  window_list = CGWindowListCopyWindowInfo(options, 0)

  for win in window_list:
    if window_name.lower() in win.get("kCGWindowName", "").lower():
      bounds = win["kCGWindowBounds"]
      return {
        "left": int(bounds["X"]),
        "top": int(bounds["Y"]),
        "width": int(bounds["Width"]),
        "height": int(bounds["Height"])
      }

  return None

def list_window_titles():
  """List all window titles on macOS"""
  options = kCGWindowListOptionOnScreenOnly
  window_list = CGWindowListCopyWindowInfo(options, 0)
  titles = []

  for win in window_list:
    title = win.get("kCGWindowName", "")
    if title:
      titles.append(title)

  return titles


def print_window_titles():
  """Print all window titles on macOS"""
  titles = list_window_titles()
  print("Available windows:")
  for title in titles:
    print(f"- {title}")
  print("")
  return titles

if __name__ == "__main__":
  print_window_titles()