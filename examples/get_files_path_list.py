import os

def get_file_path_list(path='.'):
    images = []
    for dirname, dirnames, filenames in os.walk(path):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            print(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            images += [os.path.abspath(os.path.join(dirname, filename))]

        # Advanced usage:
        # editing the 'dirnames' list will stop os.walk() from recursing into there.
        if '.git' in dirnames:
            # don't go into any .git directories.
            dirnames.remove('.git')
    return images


def main():
    images = get_file_path_list()
    for img in images:
        print(img)


if __name__ == "__main__":
    main()
