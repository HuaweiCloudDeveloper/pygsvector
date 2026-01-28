# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pygsvector'
copyright = 'Copyright (c) 2025 Huawei Technologies Co.,Ltd.'
author = 'Huawei Technologies Co.,Ltd'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',  # 添加源码链接
    'sphinx.ext.intersphinx',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

suppress_warnings = [
    'ref.ref',  # 忽略所有 "undefined label" 警告
]

# 添加项目根目录到Python路径，确保Sphinx能找到pygsvector模块
import os
import sys

# 获取当前conf.py文件所在的目录（即source目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录通常是source目录的父目录（根据你的实际结构调整..的数量）
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# 将项目根目录添加到Python解释器的搜索路径
sys.path.insert(0, project_root)

# 验证路径是否正确（可选，可注释掉）
print(f"已添加到sys.path的路径: {project_root}")