---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Computer Science IRL
This page consists of my study notes of MIT's [missing-semester](https://missing.csail.mit.edu/) course on practical CS. 

# Shell
**Concepts**:
- Shell: A textual interface & a programming environment.
- Absolute path: A path that starts with `/`.
- Relative path: A path that doesn't start with `/`.
- `$PATH`: An environment variable that lists which directories the shell should search for programs when given a command.
    - We can directly execute a program by typing its path in the command, without adding it to `$PATH`.

**Notations**:
| Notation | Description |
|:--------:|:------------|
| `$`      | NOT the root user. |
| `~`      | Root directory. |
| `/`      | Root. |
| `.`      | Current directory. |
| `..`     | Parent directory. |
| `-`/`--` | Flag. |
| `r`      | Read permission. |
| `w`      | Write permission. |
| `x`      | Execute permission. |
| `drwxr-xr--` | This is a directory.<br>The owner has `rwx` permissions.<br>The owning group has `r-x` permissions.<br>Everyone else has `r--` permissions. |
| `< file` | Read the input of the program from the file. |
| `> file` | Write the output of the program into the file. |
| `>> file` | Append the output of the program to the file. |
| `\|` | Take the output of the left as the input of the right. |
| `/sys` | A Linux-only folder with a number of kernel parameters as files. |


**Commands**:
| Command | Description |
|:-------:|:------------|
| `echo`  | Print the argument. |
| `which` | Print the file path for the given program. |
| `pwd`   | Print the current working directory. |
| `cd`    | Change directory. |
| `ls`    | Print the contents of the current directory. |
| `mv`    | Move/Rename a file. |
| `cp`    | Copy a file. |
| `rm`    | Remove a file. (`-r`: Remove a directory) |
| `mkdir` | Make a new directory. |
| `man`   | Open the manual of the given program. |
| `cat`   | Concatenate (with many functionalities). |
| `tail`  | Print the last n lines of the input. |
| `sudo`  | Do something as super user. |
| `xdg-open` | Open a file/URL. |


