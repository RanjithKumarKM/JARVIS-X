import webbrowser


def open_website(name):
    websites = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "github": "https://github.com",
        "chatgpt": "https://chatgpt.com"
    }

    if name in websites:
        print(f"Jarvis: Opening {name}...")
        webbrowser.open_new_tab(websites[name])
    else:
        print(f"Jarvis: I don't know the website '{name}' yet.")
