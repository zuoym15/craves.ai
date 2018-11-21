import webbrowser, os
class WebpageViewer:
    def open_file(self, filename):
        webbrowser.open('file://' + os.path.realpath(filename), new = 0)

    def open(self, url):
        webbrowser.open(url, new = 0)

viewer = WebpageViewer()

def task_doc():
    return {
        'actions': [
                'make -C docs/ html',
                viewer.open_file('docs/_build/html/index.html'),
                # 'open docs/_build/html/index.html',
            ],
        'verbosity': 2,
    }

def task_main():
    return {
        'actions': [
            'python HMC/ue4_to_coco.py',
            ],
        'verbosity': 2,
    }
