var=$(echo $@ | tr ' ' '\n' | grep 2 -)
matches=$(tr ' ' '\n' < test.log | grep -oP '^(.+?\/){0,2}')
echo $var
echo $matches
