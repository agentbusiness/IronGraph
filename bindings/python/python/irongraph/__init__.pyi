from os import PathLike
from typing import Any, Dict, Literal, Optional, Union

class EmbeddedDatabase:
    def __init__(
        self,
        data_dir: Union[str, PathLike[str]],
        *,
        device: Literal["auto", "cpu", "metal", "cuda"] = "auto",
        device_ordinal: int = 0,
        load_embeddings: bool = True,
        budgets: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    def query(
        self,
        cypher: str,
        *,
        project_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        query_options: Optional[Dict[str, Any]] = None,
        operation_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
    def stream_append(self, request: Dict[str, Any], *, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def stream_fetch(self, request: Dict[str, Any], *, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def status(self) -> Dict[str, Any]: ...
    def cancel(self, operation_id: str) -> bool: ...
    def snapshot(self) -> Dict[str, Any]: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> EmbeddedDatabase: ...
    def __exit__(self, exception_type: object, exception: object, traceback: object) -> bool: ...

class Client:
    @staticmethod
    def api(base_url: str) -> Client: ...
    @staticmethod
    def bolt(uri: str) -> Client: ...
    @staticmethod
    def api_mtls(
        base_url: str,
        certificate: Union[str, PathLike[str]],
        private_key: Union[str, PathLike[str]],
        certificate_authority: Union[str, PathLike[str]],
    ) -> Client: ...
    @staticmethod
    def bolt_mtls(
        uri: str,
        certificate: Union[str, PathLike[str]],
        private_key: Union[str, PathLike[str]],
        certificate_authority: Union[str, PathLike[str]],
    ) -> Client: ...
    def query(
        self,
        cypher: str,
        *,
        project_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        query_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
