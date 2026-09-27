from a2_error_taxonomy.adapters.registry import (  # noqa: F401
    available_providers,
    get_adapter_class,
    register,
)
# ★ 必须在 registry 导入之后。这一行触发 @register 执行。
from a2_error_taxonomy.adapters.DeepseekAdapter import (  # noqa: F401
    DeepseekAdapter,
)