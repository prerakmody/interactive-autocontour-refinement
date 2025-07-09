#!/usr/bin/env python3
import subprocess
import traceback
from pathlib import Path
DIR_HELPERS = Path(__file__).resolve().parent  # <root>/src/frontend/helpers
DIR_ROOT    = DIR_HELPERS.parent.parent.parent # <root>
# DIR_DIST    = DIR_ROOT / 'dist'
DIR_DIST    = DIR_HELPERS.parent / 'dist'  # <root>/src/frontend/dist


# Function to run a command and return its output
def run_command(command, split=True):
    
    try:
        commandList = command
        if split:
            commandList = command.split(' ')
        
        result = subprocess.run(commandList, shell=not split, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')

    except Exception as e:
        traceback.print_exc()   
        return 'Some error: ' + str(e)

if __name__ == '__main__':

    # Step 1 - HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" href="{pathFavicon}" type="image/png">
        <title>AI-based Interactive Refinement</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
            }}
            h1 {{
                font-size: 50px;
            }}
            p {{
                font-size: 20px;
            }}
            .command-output {{
                border: 1px solid #ccc;
                padding: 10px;
                margin-top: 20px;
                text-align: left;
                white-space: pre-wrap;
                font-family: monospace;
            }}
            h3 {{
                margin-bottom: 5px;
            }}
            h4 {{
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>502 Bad Gateway</h1>
        <p>Sorry, something went wrong. Please try again later. Contact p.p.mody@lumc.nl</p>
        <div class="command-output">
            <h3>Unix Command Outputs</h2>
            <h4>uname -a</h3><pre>{uname_output}</pre>
            <h4>df -h</h3><pre>{df_output}</pre>
            <h4>free -h</h4><pre>{memory_output}</pre>
            <h4>CPU Usage</h4><pre>{cpu_output}</pre>
            <h4>netstat -tuln | grep -E '(:50000|:55000|:8042)'</h3><pre>{netstat_output}</pre>
        </div>
    </body>
    </html>
    """

    # Step 2 - Image paths
    pathFavicon = DIR_DIST / 'favicon.png'
    if not pathFavicon.exists():
        print (" - [ERROR][502] Favicon not found at: ", pathFavicon)

    # Run commands and get their outputs
    uname_output   = run_command('uname -a')
    # df_output    = run_command('df -h')
    df_output      = run_command('df -h | grep -E "^Filesystem| /$"', split=False)
    netstat_output = run_command("netstat -tuln | grep -E '(:50000|:55000|:8042)'", split=False)
    cpu_output = run_command('top -bn1 | sed -n "7,12p"', split=False)
    memory_output = run_command('free -h')

    # Print the HTML content with the command outputs
    print("Content-type: text/html\n")
    print(html_content.format(
        pathFavicon=pathFavicon
        , uname_output=uname_output
        , df_output=df_output
        , netstat_output=netstat_output
        , cpu_output=cpu_output
        , memory_output=memory_output
    ))