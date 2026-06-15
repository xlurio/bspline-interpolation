#! /bin/bash

set -xe

if [ $# != 1 ]; then
    echo 'Usage: init_python_repo.sh {gh_user}'
fi

gh_user=$1
wget https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt -O LICENSE
wget https://github.com/github/gitignore/raw/refs/heads/main/Python.gitignore -O .gitignore
echo '# '${PWD##*/} > README.md
uv init
git init
git add .
git commit -m "first commit"
git branch -M main
gh repo create ${PWD##*/} --public
git remote add origin git@github.com:$gh_user/${PWD##*/}.git
