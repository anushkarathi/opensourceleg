import time
from abc import ABC, abstractmethod
from functools import wraps
from inspect import isabstract
from typing import Any, Callable, ClassVar, Optional, cast


class SensorNotStreamingException(Exception):
    """
    Exception raised when an operation is attempted on a sensor that is not streaming.

    This exception indicates that the sensor is not actively streaming data.
    """

    def __init__(self, sensor_name: str = "Sensor") -> None:
        """
        Initialize the SensorNotStreamingException.

        Args:
            sensor_name (str, optional): The name or identifier of the sensor.
                Defaults to "Sensor".
        """
        super().__init__(
            f"{sensor_name} is not streaming, please ensure that the connections are intact, "
            f"power is on, and the start method is called."
        )


def check_sensor_stream(func: Callable) -> Callable:
    """
    Decorator to ensure that a sensor is streaming before executing the decorated method.

    If the sensor is not streaming, a SensorNotStreamingException is raised.

    Args:
        func (Callable): The sensor method to be wrapped.

    Returns:
        Callable: The wrapped method that checks streaming status before execution.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.is_streaming:
            raise SensorNotStreamingException(sensor_name=self.__repr__())
        return func(self, *args, **kwargs)

    return wrapper


def mock_start(self: Any) -> None:
    """Mock start method."""
    self._streaming = True
    self._start_time = time.time()


def mock_stop(self: Any) -> None:
    """Mock stop method."""
    self._streaming = False


def mock_update(self: Any) -> None:
    """Mock update method that updates all properties with signal generators."""
    if not hasattr(self, "_signal_generators") or not self._signal_generators or not self._streaming:
        return

    current_time = time.time()
    if hasattr(self, "_start_time") and self._start_time is not None:
        elapsed_time = current_time - self._start_time

        for prop_name, generator in self._signal_generators.items():
            if hasattr(self, "_mock_values") and prop_name in self._mock_values:
                self._mock_values[prop_name] = generator.generate(elapsed_time)


def mock_calibrate(self: Any) -> None:
    """Mock calibrate method."""
    if hasattr(self, "_mock_values"):
        self._mock_values["is_calibrated"] = True


def mock_reset(self: Any) -> None:
    """Mock reset method."""
    if hasattr(self, "_mock_values"):
        for key in self._mock_values:
            if key != "is_calibrated":
                self._mock_values[key] = 0.0
        self._mock_values["is_calibrated"] = False


def _create_mock_is_streaming() -> property:
    """Create mock is_streaming property."""

    def mock_is_streaming(self: Any) -> bool:
        return cast(bool, self._streaming)

    return property(mock_is_streaming)


def _create_mock_data() -> property:
    """Create mock data property."""

    def mock_data(self: Any) -> Any:
        data_dict = {}
        original_class = self.__class__.__mro__[1]
        if hasattr(original_class, "ONLINE_PROPERTIES"):
            for prop_name in original_class.ONLINE_PROPERTIES:
                if prop_name not in ["data", "is_streaming"]:
                    data_dict[prop_name] = getattr(self, prop_name)
        return data_dict

    return property(mock_data)


def _create_mock_property(name: str) -> property:
    """Create mock property for generic sensor values."""

    def getter(self: Any) -> Any:
        return self._mock_values.get(name, 0.0)

    def setter(self: Any, value: Any) -> None:
        self._mock_values[name] = value

    return property(getter, setter)


def _create_mock_init(original_class: type, mock_properties: list[str]) -> Callable:
    """Create the mock __init__ method."""

    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self._streaming = False
        self._data = {}
        self._mock_values = {}
        self._start_time = None

        self._signal_generators = kwargs.pop("signal_generators", {})

        for prop_name in mock_properties:
            if hasattr(original_class, prop_name):
                attr = getattr(original_class, prop_name)
                if isinstance(attr, property):
                    self._mock_values[prop_name] = 0.0

        kwargs_copy = kwargs.copy()
        kwargs_copy["offline"] = False  # Prevent recursive mock creation

        super(type(self), self).__init__(*args, **kwargs_copy)
        self._is_offline = True

    return mock_init


def _create_mock_methods(original_class: type, mock_methods: list[str]) -> dict:
    """Create mock method implementations."""
    method_creators = {
        "start": mock_start,
        "stop": mock_stop,
        "update": mock_update,
        "calibrate": mock_calibrate,
        "reset": mock_reset,
    }

    mock_attrs = {}
    for method_name in mock_methods:
        if hasattr(original_class, method_name) and method_name in method_creators:
            mock_attrs[method_name] = method_creators[method_name]

    return mock_attrs


def _create_mock_properties(original_class: type, mock_properties: list[str]) -> dict:
    """Create mock property implementations."""
    property_creators = {
        "data": _create_mock_data,
        "is_streaming": _create_mock_is_streaming,
    }

    mock_attrs = {}
    for prop_name in mock_properties:
        if hasattr(original_class, prop_name):
            attr = getattr(original_class, prop_name)
            if isinstance(attr, property):
                if prop_name in property_creators:
                    mock_attrs[prop_name] = property_creators[prop_name]()
                else:
                    mock_attrs[prop_name] = _create_mock_property(prop_name)

    return mock_attrs


def create_mock_class(original_class: type, mock_methods: list[str], mock_properties: list[str]) -> type:
    """
    Create a mock class that inherits from the original class and overrides specified methods and properties.

    Args:
        original_class: The original sensor class
        mock_methods: List of method names to mock
        mock_properties: List of property names to mock

    Returns:
        Type: A mock class with overridden methods and properties
    """
    mock_attrs = {}
    mock_attrs["__init__"] = _create_mock_init(original_class, mock_properties)
    mock_attrs.update(_create_mock_methods(original_class, mock_methods))
    mock_attrs.update(_create_mock_properties(original_class, mock_properties))
    mock_class = type(f"Mock{original_class.__name__}", (original_class,), mock_attrs)

    return mock_class


class SensorBase(ABC):
    """
    Abstract base class for sensors.

    Defines the common interface for sensors including starting, stopping,
    updating, and streaming status.
    """

    ONLINE_METHODS: ClassVar[list[str]] = ["start", "stop", "update"]
    ONLINE_PROPERTIES: ClassVar[list[str]] = ["data", "is_streaming"]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Enforce implementation of online methods and properties in concrete subclasses.
        """
        super().__init_subclass__(**kwargs)
        # This check applies only to concrete classes that directly inherit from SensorBase
        if not isabstract(cls) and cls.__mro__[1] is SensorBase:
            if cls.ONLINE_METHODS is SensorBase.ONLINE_METHODS:
                raise NotImplementedError(
                    f"{cls.__name__} is a concrete class and must override the 'ONLINE_METHODS' class variable."
                )
            if cls.ONLINE_PROPERTIES is SensorBase.ONLINE_PROPERTIES:
                raise NotImplementedError(
                    f"{cls.__name__} is a concrete class and must override the 'ONLINE_PROPERTIES' class variable."
                )

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Create a new sensor instance, potentially replacing with a mock class for offline mode.
        """
        offline = kwargs.get("offline", False)

        if offline and hasattr(cls, "ONLINE_METHODS") and hasattr(cls, "ONLINE_PROPERTIES"):
            mock_class = create_mock_class(cls, cls.ONLINE_METHODS, cls.ONLINE_PROPERTIES)
            return mock_class.__new__(mock_class)  # type: ignore[call-overload]

        return super().__new__(cls)

    def __init__(
        self,
        tag: str,
        offline: bool = False,
        signal_generators: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the sensor base.

        Args:
            tag: Unique identifier for the sensor
            offline: Whether to run in offline mode (default: False)
            signal_generators: Dict mapping property names to SignalGenerator instances
                for generating realistic offline data (default: None)
            **kwargs: Additional keyword arguments
        """
        self._tag = tag
        self._is_offline: bool = offline
        # signal_generators will be handled by mock class if offline=True

    def __repr__(self) -> str:
        """
        Return a string representation of the sensor.

        Returns:
            str: A string identifying the sensor class.
        """
        return f"{self.tag}[{self.__class__.__name__}]"

    @property
    @abstractmethod
    def data(self) -> Any:
        """
        Get the sensor data.

        Returns:
            Any: The current data from the sensor.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        Start the sensor streaming.

        Implementations should handle initializing the sensor and beginning data acquisition.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the sensor streaming.

        Implementations should handle gracefully shutting down the sensor.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Update the sensor state or data.

        Implementations should refresh or poll the sensor data as needed.
        """
        pass

    def __enter__(self) -> "SensorBase":
        """
        Enter the runtime context for the sensor.

        This method calls start() and returns the sensor instance.

        Returns:
            SensorBase: The sensor instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exit the runtime context for the sensor.

        This method calls stop() to shut down the sensor.

        Args:
            exc_type (Any): Exception type if raised.
            exc_value (Any): Exception value if raised.
            traceback (Any): Traceback if an exception occurred.
        """
        self.stop()

    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        """
        Check if the sensor is currently streaming.

        Returns:
            bool: True if the sensor is streaming, False otherwise.
        """
        pass

    @property
    def tag(self) -> str:
        """
        Get the sensor tag.

        Returns:
            str: The unique identifier for the sensor.

        Examples:
            >>> sensor.tag
            "sensor1"
        """
        return self._tag

    @property
    def is_offline(self) -> bool:
        """
        Get the offline status of the sensor.

        Returns:
            bool: True if the sensor is offline, False otherwise.
        """
        return self._is_offline


class ADCBase(SensorBase, ABC):
    """
    Abstract base class for ADC (Analog-to-Digital Converter) sensors.

    ADC sensors are used to convert analog signals into digital data.
    """

    ONLINE_METHODS: ClassVar[list[str]] = ["start", "stop", "update", "calibrate", "reset"]
    ONLINE_PROPERTIES: ClassVar[list[str]] = ["data", "is_streaming", "is_calibrated"]

    def __init__(self, tag: str, offline: bool = False, **kwargs: Any) -> None:
        """
        Initialize the ADC sensor.
        """
        super().__init__(tag=tag, offline=offline, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the ADC sensor.

        Returns:
            str: "ADCBase"
        """
        return "ADCBase"

    def reset(self) -> None:
        """
        Reset the ADC sensor.

        Implementations should clear any stored state or calibration.
        """
        pass

    def calibrate(self) -> None:
        """
        Calibrate the ADC sensor.

        Implementations should perform necessary calibration procedures.
        """
        pass


class EncoderBase(SensorBase, ABC):
    """
    Abstract base class for encoder sensors.

    Encoders are used to measure position and velocity.
    """

    ONLINE_METHODS: ClassVar[list[str]] = ["start", "stop", "update"]
    ONLINE_PROPERTIES: ClassVar[list[str]] = ["data", "is_streaming", "position", "velocity"]

    def __init__(
        self,
        tag: str,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the encoder sensor.
        """
        super().__init__(tag=tag, offline=offline, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the encoder sensor.

        Returns:
            str: "EncoderBase"
        """
        return "EncoderBase"

    @property
    @abstractmethod
    def position(self) -> float:
        """
        Get the current encoder position.

        Returns:
            float: The current position value.
        """
        pass

    @property
    @abstractmethod
    def velocity(self) -> float:
        """
        Get the current encoder velocity.

        Returns:
            float: The current velocity value.
        """
        pass


class LoadcellBase(SensorBase, ABC):
    """
    Abstract base class for load cell sensors.

    Load cells are used to measure forces and moments.
    """

    ONLINE_METHODS: ClassVar[list[str]] = ["start", "stop", "update", "calibrate", "reset"]
    ONLINE_PROPERTIES: ClassVar[list[str]] = [
        "data",
        "is_streaming",
        "fx",
        "fy",
        "fz",
        "mx",
        "my",
        "mz",
        "is_calibrated",
    ]

    def __init__(self, tag: str, offline: bool = False, **kwargs: Any) -> None:
        """
        Initialize the load cell sensor.
        """
        super().__init__(tag=tag, offline=offline, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the load cell sensor.

        Returns:
            str: "LoadcellBase"
        """
        return "LoadcellBase"

    @abstractmethod
    def calibrate(self) -> None:
        """
        Calibrate the load cell sensor.

        Implementations should perform the calibration procedure to ensure accurate readings.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the load cell sensor.

        Implementations should reset the sensor state and any calibration data.
        """
        pass

    @property
    @abstractmethod
    def fx(self) -> float:
        """
        Get the force along the x-axis.

        Returns:
            float: The force measured along the x-axis.
        """
        pass

    @property
    @abstractmethod
    def fy(self) -> float:
        """
        Get the force along the y-axis.

        Returns:
            float: The force measured along the y-axis.
        """
        pass

    @property
    @abstractmethod
    def fz(self) -> float:
        """
        Get the force along the z-axis.

        Returns:
            float: The force measured along the z-axis.
        """
        pass

    @property
    @abstractmethod
    def mx(self) -> float:
        """
        Get the moment about the x-axis.

        Returns:
            float: The moment measured about the x-axis.
        """
        pass

    @property
    @abstractmethod
    def my(self) -> float:
        """
        Get the moment about the y-axis.

        Returns:
            float: The moment measured about the y-axis.
        """
        pass

    @property
    @abstractmethod
    def mz(self) -> float:
        """
        Get the moment about the z-axis.

        Returns:
            float: The moment measured about the z-axis.
        """
        pass

    @property
    @abstractmethod
    def is_calibrated(self) -> bool:
        """
        Check if the load cell sensor is calibrated.

        Returns:
            bool: True if calibrated, False otherwise.
        """
        pass


class IMUBase(SensorBase, ABC):
    """
    Abstract base class for Inertial Measurement Unit (IMU) sensors.

    IMUs typically provide acceleration and gyroscopic data.
    """

    ONLINE_METHODS: ClassVar[list[str]] = ["start", "stop", "update"]
    ONLINE_PROPERTIES: ClassVar[list[str]] = [
        "data",
        "is_streaming",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]

    def __init__(self, tag: str, offline: bool = False, **kwargs: Any) -> None:
        """
        Initialize the IMU sensor.
        """
        super().__init__(tag=tag, offline=offline, **kwargs)

    @property
    @abstractmethod
    def acc_x(self) -> float:
        """
        Get the estimated linear acceleration along the x-axis.

        Returns:
            float: Acceleration in m/s^2 along the x-axis.
        """
        pass

    @property
    @abstractmethod
    def acc_y(self) -> float:
        """
        Get the estimated linear acceleration along the y-axis.

        Returns:
            float: Acceleration in m/s^2 along the y-axis.
        """
        pass

    @property
    @abstractmethod
    def acc_z(self) -> float:
        """
        Get the estimated linear acceleration along the z-axis.

        Returns:
            float: Acceleration in m/s^2 along the z-axis.
        """
        pass

    @property
    @abstractmethod
    def gyro_x(self) -> float:
        """
        Get the gyroscopic measurement along the x-axis.

        Returns:
            float: Angular velocity in rad/s along the x-axis.
        """
        pass

    @property
    @abstractmethod
    def gyro_y(self) -> float:
        """
        Get the gyroscopic measurement along the y-axis.

        Returns:
            float: Angular velocity in rad/s along the y-axis.
        """
        pass

    @property
    @abstractmethod
    def gyro_z(self) -> float:
        """
        Get the gyroscopic measurement along the z-axis.

        Returns:
            float: Angular velocity in rad/s along the z-axis.
        """
        pass


if __name__ == "__main__":
    pass
