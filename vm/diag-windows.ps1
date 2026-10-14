# Test-VM diagnostics, run in the desktop session: a screenshot, and the top-level windows of a
# process with their child controls (class, text, position), via plain Win32.
#   powershell -File diag-windows.ps1 -Process VBCABLE_Setup_x64 -Out C:\ToneSphereTest\diag
param([string]$Process = 'VBCABLE_Setup_x64', [string]$Out = 'C:\ToneSphereTest\diag')
New-Item -ItemType Directory -Force $Out | Out-Null
Add-Type -Name W32 -Namespace Diag -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h); [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);'
# Bring the process's window to the front before the screenshot.
foreach ($p in @(Get-Process $Process -ErrorAction SilentlyContinue)) { if ($p.MainWindowHandle -ne [IntPtr]::Zero) { [void][Diag.W32]::ShowWindow($p.MainWindowHandle, 9); [void][Diag.W32]::SetForegroundWindow($p.MainWindowHandle) } }
Start-Sleep -Milliseconds 800
Add-Type -AssemblyName System.Drawing, System.Windows.Forms
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save((Join-Path $Out 'screen.png'))
Add-Type -TypeDefinition @'
using System; using System.Text; using System.Collections.Generic; using System.Runtime.InteropServices;
public static class Wnd {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr p, EnumProc f, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  public static string Describe(IntPtr h) {
    var c = new StringBuilder(256); var t = new StringBuilder(512); RECT r;
    GetClassName(h, c, 256); GetWindowText(h, t, 512); GetWindowRect(h, out r);
    return string.Format("{0} [{1}] '{2}' at {3},{4} {5}x{6} visible={7}", h, c, t, r.L, r.T, r.R - r.L, r.B - r.T, IsWindowVisible(h));
  }
  public static List<string> Dump(uint pid) {
    var lines = new List<string>();
    EnumWindows((h, l) => { uint p; GetWindowThreadProcessId(h, out p); if (p == pid) {
      lines.Add(Describe(h));
      EnumChildWindows(h, (c, l2) => { lines.Add("   " + Describe(c)); return true; }, IntPtr.Zero); }
      return true; }, IntPtr.Zero);
    return lines;
  }
}
'@
$lines = foreach ($p in @(Get-Process $Process -ErrorAction SilentlyContinue)) { "process $($p.Id) responding=$($p.Responding)"; [Wnd]::Dump([uint32]$p.Id) }
$lines | Set-Content (Join-Path $Out 'windows.txt')
