SPHINXOPTS=-n -W --keep-going
SOURCEDIR=source
BUILDDIR=build

.PHONY: html dirhtml json text linkcheck all clean

html:
	@sphinx-build $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

dirhtml:
	@sphinx-build $(SPHINXOPTS) -b dirhtml $(SOURCEDIR) $(BUILDDIR)/dirhtml

json:
	@sphinx-build $(SPHINXOPTS) -b json $(SOURCEDIR) $(BUILDDIR)/json

text:
	@sphinx-build $(SPHINXOPTS) -b text $(SOURCEDIR) $(BUILDDIR)/text

linkcheck:
	@sphinx-build -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

all: dirhtml json text

clean:
	rm -rf $(BUILDDIR)

