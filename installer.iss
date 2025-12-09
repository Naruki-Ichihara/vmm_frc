; Inno Setup Script for ACSC (Axial Compressive Strength Calculator)
; Requires Inno Setup 6.x: https://jrsoftware.org/isinfo.php

#define MyAppName "ACSC"
#define MyAppFullName "Axial Compressive Strength Calculator"
#define MyAppVersion "0.0.6"
#define MyAppPublisher "Naruki-Ichihara"
#define MyAppURL "https://github.com/Naruki-Ichihara/axial_compressive_strength_calculator"
#define MyAppExeName "ACSC.exe"

[Setup]
; Application information
AppId={{B8F5E3A1-2C4D-4E6F-8A0B-1C2D3E4F5A6B}
AppName={#MyAppFullName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppFullName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases

; Installation directories
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppFullName}
DisableProgramGroupPage=yes

; Output settings
OutputDir=installer_output
OutputBaseFilename=ACSC_Setup_{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes

; Visual settings
WizardStyle=modern
SetupIconFile=assets\acsc_logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; Windows version requirements
MinVersion=10.0

; Privileges
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Architecture
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; License (uncomment if you have a license file)
; LicenseFile=LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Main application files from PyInstaller output
Source: "dist\ACSC\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppFullName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppFullName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppFullName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppFullName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppFullName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
// Check for Visual C++ Redistributable (optional, may be needed for some dependencies)
function VCRedistInstalled: Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Version', Version);
end;

procedure InitializeWizard;
begin
  // Custom initialization if needed
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  // Add any pre-installation checks here
end;
