import urllib.request

link = 'https://uwmadison.app.box.com/file/986201428819'
urllib.request.urlretrieve(link, 'swin_tiny_patch4_window7_224.pth')
print('download complete')
