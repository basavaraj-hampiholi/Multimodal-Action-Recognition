# remove all .jpg files in directory/subdirectories recursively
find . -name \*.jpg -type f -delete

#remove pycache files generated during execution
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf