# How to clone the GP repo

### 1. Open the git-bash then write the command

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### 2. When prompted to "Enter a file in which to save the key," just press Enter to accept the default location.

### 3. You will be asked for a passphrase.

- Secure option: Type a password you will remember.

- Convenient option: Press Enter twice for no passphrase (useful for automated scripts, but less secure).

### 4. run the command

```bash
eval "$(ssh-agent -s)"
```

the command for powershell and cmd most likely is difrant

### 5. Add your SSH private key to the agent:

```bash ssh-add ~/.ssh/id_ed25519```

### 6. Copy the public key to your clipboard:

- Windows (Git Bash): ```bash clip < ~/.ssh/id_ed25519.pub```

- Linux: cat ```bash ~/.ssh/id_ed25519.pub``` (then manually highlight and copy the output).

### 7. Go to your Git provider:

GitHub: Settings -> SSH and GPG keys -> New SSH key.

### 8. Clone the Repository

- Finally, you can clone the repo. Ensure you use the SSH URL, not the HTTPS URL.

- Go to the repository page.

- Click the green Code button.

- Select the SSH tab (the link should start with git@github.com...).

- Run the clone command in your terminal:

```bash
git clone git@github.com:organization-name/repo-name.git
```