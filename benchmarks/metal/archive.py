"""Read-only resolution of preserved benchmark artifacts after relocation."""
import json
from pathlib import Path

from application import digest


class ArchiveResolver:
    def __init__(self, mapping):
        mapping = Path(mapping).resolve(strict=True)
        self.root = mapping.parent
        data = json.loads(mapping.read_text())
        if data.get('version') != 1 or not isinstance(data.get('paths'), dict) or not isinstance(data.get('campaigns'), dict):
            raise ValueError('archive map requires version 1, paths and campaigns')
        self.paths, self.campaigns = data['paths'], data['campaigns']
        for original, relative in self.paths.items():
            if not Path(original).is_absolute() or '..' in Path(original).parts:
                raise ValueError('archive original paths must be absolute and normalized')
            self.relative(relative)
        for relative, inventory in self.campaigns.items():
            campaign = self.relative(relative)
            if (not isinstance(inventory, dict) or not isinstance(inventory.get('files'), dict)
                    or not isinstance(inventory.get('directories'), list)
                    or 'config.json' not in inventory['files']):
                raise ValueError('archive campaign requires complete file and directory inventories')
            for name, checksum in inventory['files'].items():
                self.inventory_path(campaign, name)
                if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                    raise ValueError('archive file requires a SHA-256 digest')
            for name in inventory['directories']:
                self.inventory_path(campaign, name)
            if len(set(inventory['directories'])) != len(inventory['directories']):
                raise ValueError('duplicate archive directories')

    def inventory_path(self, campaign, name):
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or '..' in Path(name).parts or str(Path(name)) != name or name == '.'):
            raise ValueError('archive inventory requires normalized relative paths')
        return self.relative(str(campaign.relative_to(self.root) / name))

    def relative(self, value):
        if not isinstance(value, str):
            raise ValueError('archive paths must be strings')
        relative = Path(value)
        if not value or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('archive paths must be confined relative paths')
        path = self.root / relative
        self.check_tree(path)
        return path

    def check_tree(self, path):
        path = Path(path)
        if not path.is_relative_to(self.root):
            raise ValueError('artifact is outside archive root')
        # Reject even internal symlinks: inventories must describe ordinary bytes,
        # not mutable aliases. Missing case directories remain incomplete runs.
        for ancestor in (path, *path.parents):
            if ancestor == self.root:
                break
            if ancestor.is_symlink():
                raise ValueError('archive symlinks are not allowed')
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('artifact escapes archive root')
        if path.is_dir() and any(item.is_symlink() for item in path.rglob('*')):
            raise ValueError('archive symlinks are not allowed')

    def resolve(self, original):
        key = str(original)
        if key not in self.paths:
            raise ValueError('artifact has no explicit archive mapping: ' + key)
        path = self.relative(self.paths[key])
        if not path.exists():
            raise ValueError('mapped archive artifact is missing')
        return path

    def verify_config(self, path):
        self.check_tree(path.parent)
        relative = str(path.parent.relative_to(self.root))
        if relative not in self.campaigns:
            raise ValueError('campaign has no trusted archive inventory')
        expected = self.campaigns[relative]
        entries = list(path.parent.rglob('*'))
        if any(not item.is_file() and not item.is_dir() for item in entries):
            raise ValueError('archive campaign contains a nonregular entry')
        directories = sorted(str(item.relative_to(path.parent)) for item in entries if item.is_dir())
        files = {str(item.relative_to(path.parent)): digest(item) for item in entries if item.is_file()}
        if files != expected['files'] or directories != sorted(expected['directories']):
            raise ValueError('campaign does not match trusted archive inventory')
