import os
import platform
import shutil


def _is_pandoc_existed() -> bool:
    from shutil import which
    return which('pandoc') is not None


def _windows_install():
    if os.system('winget install --source winget --exact --id JohnMacFarlane.Pandoc') != 0:
        raise Exception("Failed at installing Pandoc for Windows")


def _mac_install():
    if os.system('brew install pandoc') != 0:
        raise Exception("Failed at installing Pandoc for MacOS")


def _linux_install():
    raise Exception(
        "Automatic installation is not yet available for Linux, ",
        "please follow instruction on this page to install pandoc",
        "https://pandoc.org/installing.html")


def _install_pandoc():
    print('Installing Pandoc.....')
    sysm = platform.system()
    if sysm == 'Windows':
        _windows_install()
    elif sysm == 'Darwin':
        _mac_install()
    elif sysm == 'Linux':
        _linux_install()
    else:
        raise Exception("This is an unknown operating system.")
    print('Installation Finished!')


def _check_prerequisite():
    if not _is_pandoc_existed():
        _install_pandoc()


def read_docx_table(input_filename: str, output_filename: str) -> bool:
    try:
        _check_prerequisite()
        os.system(f"pandoc -f docx -t markdown {input_filename} -o {output_filename}")
        print("Finished convert docx to md")
    except Exception as e:
        print(f"[docx2md] Exception happened during read_docx_table: {ex}")
        return False

    return True


def _example_code():
    docx_path = './docx'
    md_path = './md'

    for file_name in os.listdir(docx_path):
        input_file = os.path.join(docx_path, file_name)
        output_file = os.path.join(md_path, os.path.splitext(file_name)[0] + '.md')
        os.system(f'pandoc -f docx -t markdown {input_file} -o {output_file}')
