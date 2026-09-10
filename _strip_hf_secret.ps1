$path = Join-Path (Get-Location) 'backend/app/config.py'
if (Test-Path -LiteralPath $path) {
    $content = Get-Content -LiteralPath $path -Raw
    $content = [regex]::Replace(
        $content,
        '(?m)^(\s*HUGGINGFACE_API_KEY:\s*Optional\[str\]\s*=\s*).*$','$1None'
    )
    Set-Content -LiteralPath $path -Value $content -NoNewline
}
