#!/usr/bin/env python
# Make a html preview file for this folder

import glob

if __name__ == '__main__':
    img_files = glob.glob('*.png')
    img_tpl = '<img src="{img_link}" width="400px"></img>'

    html_content = ''

    for img_link in img_files:
        html_content += img_tpl.format(**locals())

    with open('preview.html', 'w') as f:
        f.write(html_content)

