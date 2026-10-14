# Runs inside the test VM's desktop session: plays the user's part in VB-Audio's own installer,
# which ToneSphere opens but never drives itself (VB-CABLE's licence: no silent installs).
# The installer draws its "Install Driver" button itself (no controls to automate), so it is
# clicked where it is drawn, with real mouse input; the messages after it are standard dialogs.
#   powershell -WindowStyle Hidden -File vbcable-click.ps1 [-TimeoutSeconds 300]
param([int]$TimeoutSeconds = 300)
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class Mouse {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint x, uint y, uint d, IntPtr e);
  public static void ClickAt(IntPtr h, double fx, double fy) {
    ShowWindow(h, 9); SetForegroundWindow(h); System.Threading.Thread.Sleep(500);
    RECT r; GetWindowRect(h, out r);
    SetCursorPos(r.L + (int)((r.R - r.L) * fx), r.T + (int)((r.B - r.T) * fy));
    System.Threading.Thread.Sleep(200);
    mouse_event(0x0002, 0, 0, 0, IntPtr.Zero); System.Threading.Thread.Sleep(80); mouse_event(0x0004, 0, 0, 0, IntPtr.Zero);
  }
}
'@
$ae = [System.Windows.Automation.AutomationElement]
$tree = [System.Windows.Automation.TreeScope]
$log = 'C:\ToneSphereTest\vbcable-click.log'
function Log($m) { Add-Content $log ("{0:HH:mm:ss} {1}" -f (Get-Date), $m) }
Log 'clicker started'

$clicked = $false
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
  $procs = @(Get-Process VBCABLE_Setup_x64 -ErrorAction SilentlyContinue)
  if ($procs.Count -eq 0) {
    if ($clicked) { Log 'setup closed'; break }
    Start-Sleep -Milliseconds 500; continue
  }
  $main = $procs | Where-Object { $_.MainWindowTitle -like 'VB-Audio Virtual Cable Driver Installation*' } | Select-Object -First 1
  if ($main -and -not $clicked) {
    Start-Sleep -Seconds 2  # let it finish drawing
    # "Install Driver" sits in the lower right of the 910x432 window (centre at 81 % / 87 %).
    [Mouse]::ClickAt($main.MainWindowHandle, 0.807, 0.870)
    Log "clicked Install Driver in '$($main.MainWindowTitle)'"
    $clicked = $true
    Start-Sleep -Seconds 2
    continue
  }
  # Standard dialogs from the installer (confirmation, "installation complete"): press their button.
  $ids = $procs.Id
  $wins = $ae::RootElement.FindAll($tree::Children, [System.Windows.Automation.Condition]::TrueCondition) | Where-Object { $ids -contains $_.Current.ProcessId -and $_.Current.ClassName -eq '#32770' }
  foreach ($w in $wins) {
    $text = ($w.FindAll($tree::Descendants, [System.Windows.Automation.Condition]::TrueCondition) | ForEach-Object { $_.Current.Name }) -join ' | '
    $btn = $w.FindAll($tree::Descendants, (New-Object System.Windows.Automation.PropertyCondition($ae::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button))) |
      Where-Object { $_.Current.Name -in @('OK', '&OK', 'Yes', '&Yes') } | Select-Object -First 1
    if ($btn) {
      Log "dialog '$($w.Current.Name)': $text -> pressing $($btn.Current.Name)"
      $btn.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
      Start-Sleep -Seconds 1
      # After "installation complete" the installer stays open: close it like a user would.
      if ($clicked -and $text -match 'install') {
        Start-Sleep -Seconds 2
        foreach ($p in @(Get-Process VBCABLE_Setup_x64 -ErrorAction SilentlyContinue)) { [void]$p.CloseMainWindow() }
      }
    }
  }
  Start-Sleep -Milliseconds 500
}
if (-not $clicked) { Log 'never saw the installer window' }
