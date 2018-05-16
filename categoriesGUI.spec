# -*- mode: python -*-
from distutils.sysconfig import get_python_lib
from os import path 
skimage_plugins = Tree(
    path.join(get_python_lib(), "skimage","io","_plugins"), 
    prefix=path.join("skimage","io","_plugins"),
    )

block_cipher = None


a = Analysis(['categoriesGUI.py'],
             pathex=['C:\\Users\\82641\\Desktop\\digitalMedium3_64\\categoriesGUI\\categoriesGUI'],
             binaries=[],
             datas=[],
             hiddenimports=[
                 'scipy._lib.messagestream',
                 'pywt._extensions._cwt',
                 'sklearn.neighbors.typedefs',
                 'sklearn.neighbors.quad_tree',
                 'sklearn.tree',
                 'sklearn.tree._utils'
             ],

             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='categoriesGUI',
          debug=False,
          strip=False,
          upx=True,
          console=True , icon='ico\\48.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               skimage_plugins,
               strip=False,
               upx=True,
               name='categoriesGUI')
