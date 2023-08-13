param (
    # インスタンス0のアドレス
    [Parameter(Mandatory=$true)]
    [string]$instAddress0,

    # インスタンス1のアドレス
    [Parameter(Mandatory=$true)]
    [string]$instAddress1
)

python3 run.py ${instAddress0} ${instAddress1}
