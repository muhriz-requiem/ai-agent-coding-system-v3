import os
import shutil
import tempfile
from typing import List, Tuple, Optional


class InstallTransaction:
    """Manages transactional file placements with staging, commit, and rollback capabilities."""

    def __init__(self) -> None:
        # Each staged operation will record: (src, dst, backup_path, created_parent)
        # - src: source path of the file to copy
        # - dst: target destination path
        # - backup_path: path to temporary backup if dst already existed, else None
        # - created_parent: whether the parent directory of dst was created by this op
        self._staged_ops: List[Tuple[str, str, Optional[str], bool]] = []
        self._committed = False
        self._rolled_back = False

    def stage_copy(self, src: str, dst: str) -> None:
        """Stage a file copy operation from src to dst.

        Does not immediately write to the final destination; records the action
        and handles backing up any existing target file.
        """
        if self._committed:
            raise RuntimeError("Cannot stage operations on a committed transaction.")
        if self._rolled_back:
            raise RuntimeError("Cannot stage operations on a rolled-back transaction.")

        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")

        backup_path: Optional[str] = None
        created_parent = False

        # If destination already exists, make a backup of it
        if os.path.exists(dst):
            fd, backup_path = tempfile.mkstemp(prefix="install_backup_")
            os.close(fd)
            shutil.copy2(dst, backup_path)

        self._staged_ops.append((src, dst, backup_path, created_parent))

    def commit(self) -> None:
        """Finalize installation by applying all staged file placements.

        Raises RuntimeError if any step fails, and automatically triggers rollback.
        """
        if self._committed:
            raise RuntimeError("Transaction already committed.")
        if self._rolled_back:
            raise RuntimeError("Transaction already rolled back.")

        completed_ops: List[Tuple[str, str, Optional[str], bool]] = []

        try:
            for src, dst, backup_path, _ in self._staged_ops:
                dst_dir = os.path.dirname(dst)
                created_parent = False
                if dst_dir and not os.path.exists(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    created_parent = True

                # Perform the copy using efficient chunked reading or shutil.copy2
                shutil.copy2(src, dst)
                
                # Update the completed op tuple with the parent creation flag
                completed_ops.append((src, dst, backup_path, created_parent))

            self._committed = True
        except Exception as e:
            # Roll back all successfully completed operations so far in reverse order
            self._rollback_ops(completed_ops)
            self._rolled_back = True
            raise RuntimeError(f"Commit failed, transaction rolled back: {e}") from e

    def rollback(self) -> None:
        """Reverses all staged file placements."""
        if self._rolled_back:
            return
        if self._committed:
            raise RuntimeError("Cannot roll back a committed transaction.")

        self._rollback_ops(self._staged_ops)
        self._rolled_back = True

    def _rollback_ops(self, ops: List[Tuple[str, str, Optional[str], bool]]) -> None:
        """Internal helper to reverse operations in reverse order."""
        for src, dst, backup_path, created_parent in reversed(ops):
            try:
                if backup_path and os.path.exists(backup_path):
                    # Restore from backup
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.copy2(backup_path, dst)
                    try:
                        os.remove(backup_path)
                    except OSError:
                        pass
                else:
                    # Newly created file, remove it if it exists
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)

                # Clean up parent directories if they were created and are now empty
                # (Optional cleanup, but keeps things clean)
                dst_dir = os.path.dirname(dst)
                if dst_dir and os.path.exists(dst_dir) and not os.listdir(dst_dir):
                    try:
                        os.rmdir(dst_dir)
                    except OSError:
                        pass
            except Exception:
                # Best effort cleanup during rollback
                pass
