"""
File: generate_readme.py
Author: Mark Lyngesen
Email: mark_lyngesen@live.dk
Github: https://github.com/lyngesen
Description: 

script for generating a README.md file based on the commands.tex file.
- Generate file LaTeXConfig/README.md file with all abbriviations and commands
- Generate file README.md with a main text along with all abbriviations and commands
"""

import os

def get_outstring(infile):
    with open(infile, 'r') as f: lines = f.readlines()
    
    outstring = ""

    header_row = True
    for line_number, line in enumerate(lines):
        try:
            if "%hide" in line:
                continue
            line = line.replace("%show ","")
            if "newcommand" in line or "renewcommand" in line:
                if line[0:1]== "%" not in line: continue

                if header_row:
                    outstring += "| Symbol |Shortcut|Command|Description| \n"
                    outstring += "|-----|:------|:--------|:----------| \n"

                s, description = line.split("%")
                description = description[:-1]
                new_line = s.split("ewcommand")[1]
                shortcut = s.split('{', 1)[1].split('}')[0]
                command = s.split('}', 1)[-1][1:-2]
                escaped_command = command.replace('|', '\|')
                shortcut = f"`{shortcut}`" if shortcut else ""
                if "$" not in escaped_command:
                    outstring += f"|${escaped_command}$| {shortcut} | `{escaped_command}` | {description}|\n"
                else:
                    outstring += f"|{escaped_command}| {shortcut} | `{escaped_command}` | {description}|\n"
                header_row = False
            # abbriviations
            elif "=" in line:
                abbr, term = line[2:].split(" :=")
                outstring += f"* *{abbr}* - {term}"
                header_row = True
            # remaining
            else:  
                header_row = True
                outstring += line[2:]
        except Exception as e:
            print("-"*40)
            print(f"error found on line number: {line_number+1}")
            print(f"{line=}")
            print("-"*40)
            raise e

    outstring += "\n------\n"
    outstring +="\n<!--END NOTATION-->\n"


    return outstring

    
def get_todo_list(match_str = "% TODO"):
    # run grep command
    command = "grep -H -n '% TODO' **/*.tex *.tex> todos.md"
    os.chdir("../")
    os.system(command)
    os.chdir("LaTeXConfig")

    with open("../todos.md", 'r') as f: 
        lines = f.readlines()
    outstring = ""
    outstring +="<!--START TODOS-->\n"
    outstring +="# TODO\n"
    outstring += "| TODO |LOCATION|Line| \n"
    outstring += "|-----|:------|----|\n"

    for line in lines:

        location, linenr, todo = line.split(":")[:3]
        todo = line[line.index(todo):].strip("\n").strip("% FIXME:").strip("% FIXME")
        outstring += f"|{todo}|{location}|{linenr}|\n"


    outstring +="<!--END TODOS-->\n"
    outstring += "\n------\n"
    return outstring

def append_to_file_after(outstring, outfile, START_STRING = "<!--START NOTATION-->\n"):
    """Append content of README.md file to the master README.md file"""
    
    # get string until character
    content = ""
    with open(outfile, "r") as file:
        for line in file.readlines():
            content += line
            if line == START_STRING:
                break

    with open(outfile, "w") as out: out.write(content + outstring)



def main():
    # Check script is called from dir "LaTeXConfig"
    assert os.getcwd().split("/")[-1] == "LaTeXConfig", f" script should be run in directory LaTeXConfig"

    outfile = "README.md"

    generated_message = f"`README.md` file generated from tex file `LaTeXConfig/commands.tex` using the python script `generate_readme.py`"

    # get markdown formattet version of commands.tex
    outstring = get_outstring(infile = "commands.tex")
    # save as README.md
    # with open(outfile, 'w') as out: out.write(outstring)
    append_to_file_after(outstring, outfile = "README.md")
    append_to_file_after(generated_message, outfile = "README.md", START_STRING = "<!--END NOTATION-->\n")
    # append README.md content to main README.md
    append_to_file_after(outstring, outfile = "../README.md")

    todolist = get_todo_list()
    append_to_file_after(todolist, outfile = "../README.md", START_STRING = "<!--END NOTATION-->\n")
    append_to_file_after(generated_message, outfile = "../README.md", START_STRING = "<!--END TODOS-->\n")
    # add todos list
    


if __name__ == '__main__':
    main()

