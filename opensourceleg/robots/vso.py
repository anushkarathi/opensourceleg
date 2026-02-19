import os
import time
from typing import Callable, Optional, Union

import numpy as np

from opensourceleg.actuators.base import CONTROL_MODES, ActuatorBase
from opensourceleg.actuators.maxon import DEFAULT_CURRENT_GAINS, DEFAULT_POSITION_GAINS, MaxonActuator
from opensourceleg.logging import LOGGER
from opensourceleg.robots.base import RobotBase, TActuator, TSensor
from opensourceleg.sensors.base import SensorBase
from opensourceleg.utilities import SoftRealtimeLoop


class VSO(RobotBase[TActuator, TSensor]):
    """
    Variable Stiffness Orthosis (VSO) class derived from RobotBase.
    """

    def start(self) -> None:
        """
        Start the VSO.
        """
        super().start()

    def stop(self) -> None:
        """
        Stop the VSO.
        """
        super().stop()

    def update(self) -> None:
        """
        Update the VSO.
        """
        super().update()

    def home(
        self,
        homing_voltage: int = 2000,
        homing_frequency: int = 40,
        homing_direction: Optional[dict[str, int]] = None,
        output_position_offset: Optional[dict[str, float]] = None,
        current_threshold: int = 5000,
        velocity_threshold: float = 0.001,
        callbacks: Optional[dict[str, Callable]] = None,
    ) -> None:
        """
        Call the home method for all actuators.

        Args:
            homing_voltage: The voltage to apply to the actuators during homing.
            homing_frequency: The frequency to apply to the actuators during homing.
            homing_direction: The direction to apply to the actuators during homing.
                Default is -1 for knee and ankle.
            output_position_offset: The offset to apply to the actuators during homing.
                Default is 0.0 for knee and 30.0 for ankle.
            current_threshold: The current threshold to apply to the actuators during homing. Default is 5000.
            velocity_threshold: The velocity threshold to apply to the actuators during homing. Default is 0.001.
            callbacks  Optional[dict[str, Callable]]:
                Optional dictionary of callback functions, one per actuator, to be called when each actuator's
                homing completes. Only one callback per actuator is supported, and the tag must match.
                Each function should take no arguments and return None. If None, no callbacks are used.
        """
        if output_position_offset is None:
            output_position_offset = {"ankle": np.deg2rad(30.0)}
        if homing_direction is None:
            homing_direction = {"ankle": -1}
        for actuator in self.actuators.values():
            callback = callbacks.get(actuator.tag, None) if callbacks is not None else None
            actuator.home(
                homing_voltage=homing_voltage,
                homing_frequency=homing_frequency,
                homing_direction=homing_direction[actuator.tag],
                output_position_offset=output_position_offset[actuator.tag],
                current_threshold=current_threshold,
                velocity_threshold=velocity_threshold,
                callback=callback,
            )

        LOGGER.info(
            "OSL homing complete. If you'd like to create or load encoder maps to "
            "correct for nonlinearities, call `make_encoder_linearization_map()` method."
        )

    def make_encoder_linearization_map(
        self,
        overwrite: bool = False,
    ) -> None:
        """
        This method makes a lookup table to calculate the position measured by the joint encoder.
        This is necessary because the magnetic output encoders are nonlinear.
        By making the map while the joint is unloaded, joint position calculated by motor position * gear ratio
        should be the same as the true joint position. Output from this function is a file containing a_i values
        parameterizing the map.

        Eqn:
            position = sum from i=0^5 (a_i*counts^i)

        Author:
            Kevin Best (tkbest@umich.edu),
            Senthur Ayyappan (senthura@umich.edu)
        """
        for actuator_key in self.actuators:
            if f"joint_encoder_{actuator_key}" in self.sensors:
                self._create_linear_joint_mapping(
                    actuator_key=actuator_key,
                    encoder_key=f"joint_encoder_{actuator_key}",
                    overwrite=overwrite,
                )
            else:
                LOGGER.warning(
                    f"[{actuator_key}] No joint encoder found. Skipping. "
                    f"Encoder tags should be of the form 'joint_encoder_{actuator_key}'."
                )

    def _create_linear_joint_mapping(
        self,
        actuator_key: str,
        encoder_key: str,
        overwrite: bool = False,
    ) -> None:
        _actuator: ActuatorBase = self.actuators[actuator_key]
        _encoder: SensorBase = self.sensors[encoder_key]

        if not _actuator.is_homed:
            LOGGER.warning(
                msg=f"[{str.upper(_actuator.tag)}] Please home the {_actuator.tag} joint before making the encoder map."
            )
            return None

        if os.path.exists(f"./{_encoder.tag}_linearization_map.npy") and not overwrite:
            LOGGER.info(msg=f"[{str.upper(_encoder.tag)}] Encoder map exists. Skipping encoder map creation.")
            _encoder.set_encoder_map(  # type: ignore[attr-defined]
                np.polynomial.polynomial.Polynomial(np.load(f"./{_encoder.tag}_linearization_map.npy"))
            )
            LOGGER.info(
                msg=f"[{str.upper(_encoder.tag)}] Encoder map loaded from './{_encoder.tag}_linearization_map.npy'."
            )
            return None

        _actuator.set_control_mode(mode=CONTROL_MODES.CURRENT)
        _actuator.set_current_gains(
            kp=DEFAULT_CURRENT_GAINS.kp,
            ki=DEFAULT_CURRENT_GAINS.ki,
            kd=DEFAULT_CURRENT_GAINS.kd,
            ff=DEFAULT_CURRENT_GAINS.ff,
        )

        time.sleep(0.1)

        _actuator.set_output_torque(value=0.0)

        _joint_encoder_array = []
        _output_position_array = []

        LOGGER.info(
            msg=f"[{str.upper(_actuator.tag)}] Please manually move the {_actuator.tag} joint numerous times through "
            f"its full range of motion for 10 seconds."
        )
        input("Press any key when you are ready to start.")

        _start_time: float = time.time()

        # TODO: Switch to SoftRealtimeLoop since it has reset method now
        while time.time() - _start_time < 10:
            try:
                LOGGER.info(
                    msg=f"[{str.upper(_actuator.tag)}] Mapping the {_actuator.tag} "
                    f"joint encoder: {(10 - time.time() + _start_time):.2f} seconds left."
                )
                _actuator.update()
                _encoder.update()

                _joint_encoder_array.append(_encoder.position)  # type: ignore[attr-defined]
                _output_position_array.append(_actuator.output_position)
                time.sleep(1 / _actuator.frequency)

            except KeyboardInterrupt:
                LOGGER.warning(msg="Encoder map interrupted.")
                return None

        LOGGER.info(msg=f"[{str.upper(_actuator.tag)}] You may now stop moving the {_actuator.tag} joint.")

        _power = np.arange(4.0)
        _a_mat = np.array(_joint_encoder_array).reshape(-1, 1) ** _power
        _beta = np.linalg.lstsq(_a_mat, _output_position_array, rcond=None)
        _coeffs = _beta[0]

        _encoder.set_encoder_map(np.polynomial.polynomial.Polynomial(coef=_coeffs))  # type: ignore[attr-defined]

        np.save(file=f"./{_encoder.tag}_linearization_map.npy", arr=_coeffs)

        _actuator.set_control_mode(mode=CONTROL_MODES.VOLTAGE)
        _actuator.set_motor_voltage(value=0.0)

        LOGGER.info(
            msg=f"[{str.upper(_encoder.tag)}] Encoder map saved to './{_encoder.tag}_linearization_map.npy' and loaded."
        )

    @property
    def ankle(self) -> Union[TActuator, ActuatorBase]:
        """
        Get the ankle actuator.

        Returns:
            Union[TActuator, ActuatorBase]: The ankle actuator.
        """
        try:
            return self.actuators["ankle"]
        except KeyError:
            LOGGER.error("Ankle actuator not found. Please check for `ankle` key in the actuators dictionary.")
            exit(1)

    @property
    def joint_encoder_ankle(self) -> Union[TSensor, SensorBase]:
        """
        Get the ankle joint encoder sensor.

        Returns:
            Union[TSensor, SensorBase]: The ankle joint encoder sensor.
        """
        try:
            return self.sensors["joint_encoder_ankle"]
        except KeyError:
            LOGGER.error(
                "Ankle joint encoder sensor not found."
                "Please check for `joint_encoder_ankle` key in the sensors dictionary."
            )
            exit(1)


if __name__ == "__main__":
    frequency = 200

    vso = VSO[MaxonActuator, SensorBase](
        tag="VariableStiffnessOrthosis",
        actuators={
            "ankle": MaxonActuator("ankle", offline=False, frequency=frequency, gear_ratio=1),
        },
        sensors={
            "encoder": SensorBase(),
        },
    )

    with vso:
        vso.update()

        vso.ankle.set_control_mode(CONTROL_MODES.POSITION)
        vso.ankle.set_position_gains(
            kp=DEFAULT_POSITION_GAINS.kp,
            ki=DEFAULT_POSITION_GAINS.ki,
            kd=DEFAULT_POSITION_GAINS.kd,
            ff=DEFAULT_POSITION_GAINS.ff,
        )
        vso.ankle.set_output_position(vso.ankle.output_position + np.deg2rad(10))

        while True:
            try:
                vso.update()
                # print(osl.sensors["loadcell"].fz)
                time.sleep(1 / frequency)

            except KeyboardInterrupt:
                exit()
