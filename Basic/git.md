remove a large file from git log
```
git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch <file/dir>' HEAD
```
where `<file/dir>` should be replaced by a real file/dir name