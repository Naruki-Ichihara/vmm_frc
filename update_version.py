"""Update version in installer.iss from vmm.__version__"""
import re
import sys
from vmm import __version__

def update_installer_version():
    """Update the version in installer.iss to match vmm.__version__"""
    try:
        # Read current installer.iss
        with open('installer.iss', 'r', encoding='utf-8') as f:
            content = f.read()

        # Update version
        new_content = re.sub(
            r'#define MyAppVersion "[^"]+"',
            f'#define MyAppVersion "{__version__}"',
            content
        )

        # Write back
        with open('installer.iss', 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Verify
        with open('installer.iss', 'r', encoding='utf-8') as f:
            verify_content = f.read()
            if f'#define MyAppVersion "{__version__}"' in verify_content:
                print(f'Successfully updated installer.iss to version {__version__}')
                return 0
            else:
                print(f'ERROR: Version update verification failed', file=sys.stderr)
                return 1

    except Exception as e:
        print(f'ERROR: Failed to update installer.iss: {e}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(update_installer_version())
