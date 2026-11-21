param(
    [Parameter(Mandatory=$true)][string]$Bundle,
    [string]$SshAlias
)
$ErrorActionPreference = 'Stop'
$SshConfigPath = Join-Path $env:USERPROFILE '.ssh/config'
if (-not (Test-Path -LiteralPath $SshConfigPath)) { throw 'No existing SSH config; no hostname is inferred.' }
$ConfiguredAliases = @(Get-Content -LiteralPath $SshConfigPath | ForEach-Object {
    if ($_ -match '^\s*Host\s+(.+)$') { $Matches[1] -split '\s+' | Where-Object { $_ -notmatch '[*?!]' } }
})
if (-not $SshAlias) {
    $Candidates = @($ConfiguredAliases | Where-Object { $_ -match 'hypatia' })
    if ($Candidates.Count -ne 1) { throw 'Select an existing SSH alias with -SshAlias; no credentials or hostnames are guessed.' }
    $SshAlias = $Candidates[0]
}
if ($SshAlias -notin $ConfiguredAliases -or $SshAlias -notmatch '^[a-zA-Z0-9_.-]+$') { throw 'SSH alias must exist in your config.' }
$BundlePath = (Resolve-Path -LiteralPath $Bundle).Path
$ChecksumPath = $BundlePath + '.sha256'
if (-not (Test-Path -LiteralPath $ChecksumPath)) { throw 'Missing bundle checksum file.' }
$ExpectedHash = (Get-Content -LiteralPath $ChecksumPath -Raw).Split(' ')[0].Trim()
if ((Get-FileHash -LiteralPath $BundlePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) { throw 'Bundle checksum mismatch.' }
# Transfer only; does not execute remote commands or submit GPU jobs.
& scp -- $BundlePath $ChecksumPath "${SshAlias}:"
if ($LASTEXITCODE -ne 0) { throw 'scp failed; original bundle retained.' }
