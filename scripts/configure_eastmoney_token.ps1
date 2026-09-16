[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$variableName = "QUANTPAPER_EASTMONEY_TOKEN"
$script:tokenValue = $null

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = "Eastmoney Token Setup"
$form.StartPosition = "CenterScreen"
$form.Width = 560
$form.Height = 190
$form.TopMost = $true

$label = New-Object System.Windows.Forms.Label
$label.AutoSize = $true
$label.Left = 20
$label.Top = 20
$label.Text = "Paste the new Eastmoney quant token below. The input is hidden."

$textBox = New-Object System.Windows.Forms.TextBox
$textBox.Left = 20
$textBox.Top = 52
$textBox.Width = 500
$textBox.UseSystemPasswordChar = $true

$saveButton = New-Object System.Windows.Forms.Button
$saveButton.Left = 330
$saveButton.Top = 92
$saveButton.Width = 90
$saveButton.Text = "Save"

$cancelButton = New-Object System.Windows.Forms.Button
$cancelButton.Left = 430
$cancelButton.Top = 92
$cancelButton.Width = 90
$cancelButton.Text = "Cancel"

$saveButton.Add_Click({
    $value = $textBox.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($value)) {
        [System.Windows.Forms.MessageBox]::Show("Token cannot be empty.", "Eastmoney Token") | Out-Null
        return
    }
    if ($value -match '[\x00-\x1F\x7F]') {
        [System.Windows.Forms.MessageBox]::Show("The token contains control characters. Copy it again as plain text.", "Eastmoney Token") | Out-Null
        return
    }
    $script:tokenValue = $value
    $form.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Close()
})

$cancelButton.Add_Click({
    $form.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Close()
})

$form.AcceptButton = $saveButton
$form.CancelButton = $cancelButton
$form.Controls.Add($label)
$form.Controls.Add($textBox)
$form.Controls.Add($saveButton)
$form.Controls.Add($cancelButton)
$textBox.Focus()

$result = $form.ShowDialog()
if ($result -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Host "Cancelled. The token was not changed."
    exit 0
}

[Environment]::SetEnvironmentVariable($variableName, $script:tokenValue, "User")
$textBox.Clear()
$script:tokenValue = $null

Write-Host "Token saved to the current Windows user environment."
Write-Host "Close and reopen the terminal before starting the quant runner."
