# PowerShell script to create desktop shortcut for Marelli AI Inspection System
# Run this as Administrator in PowerShell

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "         MARELLI AI SHORTCUT CREATOR" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Get the project path (hardcoded to your current location)
$projectPath = "C:\Users\manis\Downloads\Final Ops Code\Final Ops Code"
$desktopPath = [Environment]::GetFolderPath("Desktop")

# Define file paths
$batFilePath = "$projectPath\start_marelli_server.bat"
$shortcutPath = "$desktopPath\Marelli AI Inspector.lnk"
$iconPath = "$projectPath\renata_logo.ico"

Write-Host "Project Path: $projectPath" -ForegroundColor Yellow
Write-Host "Desktop Path: $desktopPath" -ForegroundColor Yellow
Write-Host ""

# Check if batch file exists
if (-not (Test-Path $batFilePath)) {
    Write-Host "ERROR: start_marelli_server.bat not found!" -ForegroundColor Red
    Write-Host "Please ensure the batch file exists in: $batFilePath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create shortcut
Write-Host "Creating desktop shortcut..." -ForegroundColor Green
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $batFilePath
$Shortcut.WorkingDirectory = $projectPath
$Shortcut.Description = "Marelli AI Industrial Nut Detection System"
$Shortcut.WindowStyle = 1  # Normal window

# Set icon
if (Test-Path $iconPath) {
    $Shortcut.IconLocation = $iconPath
    Write-Host "Using custom icon: renata_logo.ico" -ForegroundColor Green
} else {
    # Use a professional system icon for industrial applications
    $Shortcut.IconLocation = "C:\Windows\System32\shell32.dll,109"  # Gear/settings icon
    Write-Host "Custom icon not found. Using system gear icon." -ForegroundColor Yellow
    Write-Host "To use custom icon, place 'renata_logo.ico' in project folder" -ForegroundColor Yellow
}

# Savetheshortcut
$Shortcut.Save()

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "SUCCESS! Desktop shortcut created!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Shortcut Name: Marelli AI Inspector" -ForegroundColor White
Write-Host "Location: $shortcutPath" -ForegroundColor White
Write-Host ""
Write-Host "Usage Instructions:" -ForegroundColor Cyan
Write-Host "  1. Double-click the desktop shortcut" -ForegroundColor White
Write-Host "  2. Wait for server to start (about 10-15 seconds)" -ForegroundColor White
Write-Host "  3. Browser will automatically open to the application" -ForegroundColor White
Write-Host "  4. Press Ctrl+C in the command window to stop server" -ForegroundColor White
Write-Host ""
Write-Host "The shortcut is ready to use!" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to exit"