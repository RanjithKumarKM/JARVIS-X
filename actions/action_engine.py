import webbrowser
import subprocess
import os


# ============================================================
# WEBSITE CONFIGURATION
# ============================================================

WEBSITES = {

    "youtube":
        "https://www.youtube.com",

    "google":
        "https://www.google.com",

    "github":
        "https://github.com",

    "chatgpt":
        "https://chatgpt.com",

    "reddit":
        "https://www.reddit.com",

    "linkedin":
        "https://www.linkedin.com"
}


# ============================================================
# OPEN WEBSITE
# ============================================================

def open_website(command):

    command = command.lower()

    for website, url in WEBSITES.items():

        if website in command:

            print(
                f"🌐 Opening {website}..."
            )

            webbrowser.open(url)

            return True

    print(
        "❌ I couldn't identify the website."
    )

    return False


# ============================================================
# OPEN APPLICATION
# ============================================================

def open_application(command):

    command = command.lower()

    # --------------------------------------------
    # Windows applications
    # --------------------------------------------

    if "notepad" in command:

        print("💻 Opening Notepad...")

        subprocess.Popen(
            ["notepad.exe"]
        )

        return True

    if "calculator" in command:

        print("💻 Opening Calculator...")

        subprocess.Popen(
            ["calc.exe"]
        )

        return True
    if "microsoft edge" in command or "edge" in command:

        print("💻 Opening Microsoft Edge...")

        edge_paths = [
            os.path.expandvars(
                r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
            ),
            os.path.expandvars(
                r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
            ),
        ]

        for path in edge_paths:

            if os.path.exists(path):
                subprocess.Popen([path])
                return True

        print("❌ Microsoft Edge executable not found.")

        return False

    if "paint" in command:

        print("💻 Opening Paint...")

        subprocess.Popen(
            ["mspaint.exe"]
        )

        return True

    if "file explorer" in command:

        print("💻 Opening File Explorer...")

        subprocess.Popen(
            ["explorer.exe"]
        )

        return True

    # --------------------------------------------
    # Visual Studio Code
    # --------------------------------------------

    if (
        "visual studio code" in command
        or "vs code" in command
        or "vscode" in command
    ):

        print(
            "💻 Opening Visual Studio Code..."
        )

        try:

            subprocess.Popen(
                ["code"]
            )

        except FileNotFoundError:

            print(
                "❌ VS Code command was not found."
            )

            return False

        return True

    print(
        "❌ I couldn't identify the application."
    )

    return False


# ============================================================
# ACTION DISPATCHER
# ============================================================

def execute_action(
    intent,
    command
):

    if intent == "OPEN_WEBSITE":

        return open_website(
            command
        )

    elif intent == "OPEN_APPLICATION":

        return open_application(
            command
        )

    else:

        print(
            f"⚠️ No action implemented yet "
            f"for intent: {intent}"
        )

        return False
