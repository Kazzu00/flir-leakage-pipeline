param(
    [Parameter(Mandatory=$true)][string]$PackageRoot,
    [Parameter(Mandatory=$true)][string]$SshTarget,
    [Parameter(Mandatory=$true)][string]$RemoteDirectory
)
$ErrorActionPreference = 'Stop'
if ($SshTarget -notmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+$') { throw 'Use user@hostname' }
if ($RemoteDirectory -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteDirectory.Contains('..')) {
    throw 'Use an absolute remote directory without spaces or traversal'
}
$archive = Join-Path $PackageRoot 'base_hud.tar'
$checksums = Join-Path $PackageRoot 'SHA256SUMS'
if (!(Test-Path -LiteralPath $archive) -or !(Test-Path -LiteralPath $checksums)) {
    throw 'Missing prepared archive/checksums'
}
# The user creates the destination manually; no SSH provisioning is performed here.
scp -- $archive $checksums "${SshTarget}:${RemoteDirectory}/"
if ($LASTEXITCODE -ne 0) { throw 'Transfer failed' }
