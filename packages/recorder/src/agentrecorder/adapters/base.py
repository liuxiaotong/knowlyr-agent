"""适配器基类定义."""

from abc import ABC, abstractmethod

from agentrecorder.schema import Trajectory


class BaseAdapter(ABC):
    """Agent 框架适配器基类.

    所有适配器必须继承此类，并实现 parse() 和 validate() 方法。

    实现新适配器时：
    1. 继承 BaseAdapter
    2. 实现 parse() 将日志转换为 Trajectory
    3. 实现 validate() 验证日志格式
    4. 在 adapters/__init__.py 中注册
    """

    @abstractmethod
    def parse(self, log_path: str) -> Trajectory:
        """将 Agent 日志解析为标准轨迹格式.

        Args:
            log_path: 日志文件路径。

        Returns:
            标准化轨迹对象。
        """
        ...

    @abstractmethod
    def validate(self, log_path: str) -> bool:
        """验证日志文件是否为该框架格式.

        Args:
            log_path: 日志文件路径。

        Returns:
            如果日志格式匹配该适配器则返回 True。
        """
        ...
