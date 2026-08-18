from actions.browser_actions import open_website


def main():
    command = input("You: ").lower().strip()

    if command.startswith("open "):
        website = command[5:]
        open_website(website)

    else:
        print("Jarvis: I don't understand that command yet.")


if __name__ == "__main__":
    main()
