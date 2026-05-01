param(
    [Parameter(Mandatory=$true)]
    [string]$Version,
    [switch]$SyncObsidian
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Set-Location $repoRoot

# ---------- validation ----------
if ($Version -notmatch '^\d+\.\d+\.\d+$') {
    Write-Error "Invalid version: $Version. Expected semver, e.g. 0.4.2"
    exit 1
}

# ---------- bump ----------
Write-Host "=== Bump version to $Version ==="

# pyproject.toml
$pyproject = Get-Content "pyproject.toml" -Raw
$pyproject = $pyproject -replace '(?<=^version\s*=\s*")[^"]+', $Version
$pyproject | Set-Content "pyproject.toml" -NoNewline
Write-Host "  pyproject.toml -> $Version"

# skill SKILL.md
$skillMd = Get-Content "docs\skills\voiceover-pipeline\SKILL.md" -Raw
$skillMd = $skillMd -replace '(?<=voiceover-pipeline\s+)[\d.]+', $Version
$skillMd | Set-Content "docs\skills\voiceover-pipeline\SKILL.md" -NoNewline
Write-Host "  SKILL.md -> $Version"

# skill 00-version-log.md
$verLog = Get-Content "docs\skills\voiceover-pipeline\docs\00-version-log.md" -Raw
$verLog = $verLog -replace '(?<=\*\*Compatible app\*\* \| voiceover-pipeline\s+)[\d.]+', $Version
$verLog = $verLog -replace '(?<=\*\*Максимальная проверенная\*\* \| )[\d.]+', $Version
$verLog | Set-Content "docs\skills\voiceover-pipeline\docs\00-version-log.md" -NoNewline
Write-Host "  version-log.md -> $Version"

# ---------- gate ----------
Write-Host "=== Gate: pytest ==="
$pytest = uv run pytest --tb=short -q 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Gate failed: pytest"
    Write-Host $pytest
    exit 1
}
$passed = ($pytest -match '(\d+) passed').Matches.Groups[1].Value
Write-Host "  $passed passed"

Write-Host "=== Gate: list providers ==="
$providers = uv run voiceover list providers --json 2>&1 | ConvertFrom-Json
if ($providers.status -ne "success") { Write-Error "Gate failed: list providers"; exit 1 }
Write-Host "  $($providers.providers.Count) providers"

Write-Host "=== Gate: list voices ==="
$polzaVoices = uv run voiceover list voices --provider polza-tts --json 2>&1 | ConvertFrom-Json
$orVoices = uv run voiceover list voices --provider openrouter-tts --json 2>&1 | ConvertFrom-Json
if ($polzaVoices.status -ne "success" -or $orVoices.status -ne "success") {
    Write-Error "Gate failed: list voices"
    exit 1
}
Write-Host "  polza-tts: $($polzaVoices.voices.Count) voices, openrouter-tts: $($orVoices.voices.Count) voices"

# ---------- build ----------
Write-Host "=== Build ==="
Remove-Item dist\* -Force -ErrorAction SilentlyContinue
uv build 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

$sdist = Get-ChildItem "dist\voiceover_pipeline-$Version.tar.gz" -ErrorAction Stop
$wheel = Get-ChildItem "dist\voiceover_pipeline-$Version-py3-none-any.whl" -ErrorAction Stop
Write-Host "  $($sdist.Name)"
Write-Host "  $($wheel.Name)"

# ---------- package skill ----------
Write-Host "=== Package skill ==="
$skillBuilderPackageScript = "$env:USERPROFILE\.config\opencode\skills\skill-builder\scripts\package_skill.py"
python $skillBuilderPackageScript "docs\skills\voiceover-pipeline" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "Skill packaging failed"; exit 1 }

$skillFile = "dist\voiceover-pipeline-$Version.skill"
Move-Item -Force "docs\skills\voiceover-pipeline.skill" $skillFile -ErrorAction Stop
$skillSize = [math]::Round((Get-Item $skillFile).Length / 1KB)
Write-Host "  voiceover-pipeline-$Version.skill ($skillSize KB)"

# ---------- stage + commit + tag ----------
Write-Host "=== Commit ==="
$stageFiles = @(
    "pyproject.toml",
    "docs\skills\voiceover-pipeline\SKILL.md",
    "docs\skills\voiceover-pipeline\docs\00-version-log.md"
)
foreach ($f in $stageFiles) {
    if (Test-Path $f) { git add $f 2>&1 | Out-Null }
}

# also stage any other modified files the user may want
# (git status for user review)

git commit -m "Release voiceover-pipeline $Version" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "Commit failed"; exit 1 }
Write-Host "  committed"

Write-Host "=== Tag v$Version ==="
git tag v$Version 2>&1 | Out-Null
git push 2>&1 | Out-Null
git push --tags 2>&1 | Out-Null
Write-Host "  pushed v$Version"

# ---------- publish PyPI ----------
Write-Host "=== Publish PyPI ==="
$envContent = Get-Content ".env" -Raw -ErrorAction Stop
$match = [regex]::Match($envContent, 'PYPI_TOKEN=(.*)')
if (-not $match.Success) {
    Write-Error "PYPI_TOKEN not found in .env"
    exit 1
}
$token = $match.Groups[1].Value.Trim()

uv publish --token $token $sdist.FullName $wheel.FullName 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "Publish failed"; exit 1 }
Write-Host "  published voiceover-pipeline $Version to PyPI"

# ---------- optional: sync to Obsidian ----------
if ($SyncObsidian) {
    Write-Host "=== Sync skill to Obsidian ==="
    $obsidianSkill = "E:\AUTO_OBSIDIAN\.opencode\skills\voiceover-pipeline"
    Remove-Item -Recurse -Force $obsidianSkill -ErrorAction SilentlyContinue
    Copy-Item -Recurse "docs\skills\voiceover-pipeline" $obsidianSkill
    Write-Host "  synced to $obsidianSkill"
}

Write-Host "=== Done ==="
Write-Host "GitHub Release: attach $($sdist.Name), $($wheel.Name), $($skillFile | Split-Path -Leaf)"
