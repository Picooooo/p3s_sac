from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.distributions import bijector
from tensorflow.python.ops.distributions import util as distribution_util


__all__ = ["ConditionalBijector"]


class ConditionalBijector(bijector.Bijector):
  """Conditional Bijector is a Bijector that allows intrinsic conditioning."""

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def forward(self, x, name="forward", **condition_kwargs):
    return self._call_forward(x, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def inverse(self, y, name="inverse", **condition_kwargs):
    return self._call_inverse(y, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def inverse_log_det_jacobian(
      self, y, name="inverse_log_det_jacobian", **condition_kwargs):
    return self._call_inverse_log_det_jacobian(y, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def forward_log_det_jacobian(
      self, x, name="forward_log_det_jacobian", **condition_kwargs):
    return self._call_forward_log_det_jacobian(x, name, **condition_kwargs)