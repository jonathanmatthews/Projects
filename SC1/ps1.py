# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:12:38 2018

@author: Jonathan Matthews (jxm1027)

A set of 4 classes (Vector (of length 4), 4x4 Matrix, FourVector, and BoostMatrix) to
be used in particle physics computations.
"""

class Vector:
    """A (length 4) Vector class, with support for complex elements. Problem 1."""

    def __init__(self, a, b, c, d):
        """The constructor for the Vector class. Each element of the Vector should be
           provided as a separate argument to the constructor."""
        self.components = [a, b, c, d]

    def __str__(self):
        """Returns a string representation of an instance of Vector."""
        return "[" + " ".join(str(i) for i in self.components) + "]"

    def __repr__(self):
        """Returns a string which can be used to reproduce an instance of Vector."""
        return "Vector(" + ", ".join(str(i) for i in self.components) + ")"

    def __getitem__(self, i):
        """Returns the element at index 'i'."""
        return self.components[i]

    def __setitem__(self, i, val):
        """Sets the element at index 'i' to the value of 'val'."""
        self.components[i] = val

    def __len__(self):
        """Returns the number of elements in an instance of the Vector class."""
        return 4

    def __add__(self, vec):
        """Returns the sum of two Vectors, self and 'vec'."""
        result = self.__class__(0, 0, 0, 0)
        for i in range(4):
            result[i] = self[i] + vec[i]
        return result

    def __iadd__(self, vec):
        """Augmented assignment for addition of two Vectors."""
        self = self + vec
        return self

    def __sub__(self, vec):
        """ Returns the result of subtracting 'vec' (a Vector) from self."""
        result = self.__class__(0, 0, 0, 0)
        for i in range(4):
            result[i] = self[i] - vec[i]
        return result

    def __isub__(self, vec):
        """Augmented assignment for subtraction of Vector 'vec' from the Vector being
           assigned to."""
        self = self - vec
        return self

    def __pos__(self):
        """Implementation for the unary '+' operator. Returns the instance of
           Vector with no modification."""
        return self

    def __neg__(self):
        """Implementation of the unary '-' operator. Returns the instance of
           Vector with all element signs flipped."""
        return self.__class__(0, 0, 0, 0) - self

    def __invert__(self):
        """Implementation of the unary '~' operator. Returns a Vector containing
           the conjugate of all elements of the Vector the operator was applied to."""
        result = Vector(0, 0, 0, 0)
        for i in range(4):
            result[i] = self[i].conjugate()
        return result

    def __mul__(self, other):
        """Returns the result of an instance of Vector being multiplied by 'other'.
           'other' may be a scalar, Vector or Matrix.

           If it is a scalar, each element of the Vector will be multiplied by 'other'.
           If it is a Vector, the product of 'other' and the transposition of self will
           be returned, as a scalar.
           If it is a Matrix, the product of 'other' and the transposition of self will
           be returned, as a Vector or an inherited class."""

        result = self.__class__(0, 0, 0, 0)
        try:
            for i in range(4):
                result[i] = self[i]*other[i] # If other is vector.
            return sum(result.components)
        except:
            if isinstance(other, Matrix): # If other is Matrix.
                return other.__rmul__(self)
            for i in range(4):
                result[i] = self[i]*other # Assume other is scalar.
            return result

    def __rmul__(self, other):
        """Returns the reverse multiplication of an instance of Vector and 'other'.
           Follows same rules as __mul__, except with the roles of self and 'other'
           being swapped (in the case of non-commutative multiplication)."""
        return self*other

    def __imul__(self, other):
        """Augmented assignment for multiplication of Vector with 'other'. Follows same
           rules as __mul__."""
        self = self*other
        return self

    def __truediv__(self, other):
        """Returns the result of dividing an instance of Vector by a scalar, 'other'."""
        result = self.__class__(0, 0, 0, 0)
        for i in range(4):
            result[i] = self[i]/other # Scalar 'other' only.
        return result

    def __idiv__(self, other):
        """Augmented assignment for division of a Vector by a scalar, 'other'."""
        self = self/other
        return self

    def __abs__(self):
        """Returns the Frobenius norm of a Vector instance."""
        return (sum(map(lambda x: x*x.conjugate(), self))**0.5).real

    def __pow__(self, power):
        """Returns the value of the Frobenius norm raised to the given integer
           power 'power', if power is even, or power-1 is power is odd."""
        return abs(self)**((power//2)*2) # (power//2)*2 = power if even, power-1 if odd.

    def __ipow__(self, power):
        """Augmented assignment for exponentiation of Vectors, follows __pow__ rules."""
        self = self**power
        return self



class Matrix:
    """A 4x4 Matrix class, with support for complex elements. Problem 2."""

    def __init__(self, a, b, c, d):
        """The constructor for the Matrix class. Each of the 4 rows should be
           given as an iterable of length 4, as a separate argument."""
        self.components = [[a[0], a[1], a[2], a[3]],\
                           [b[0], b[1], b[2], b[3]],\
                           [c[0], c[1], c[2], c[3]],\
                           [d[0], d[1], d[2], d[3]]]

    def __str__(self):
        """Returns a string representation of an instance of Matrix."""
        return "[[" + "]\n [".\
        join(" ".join(str(j) for j in self.components[i]) for i in range(4)) + "]]"

    def __repr__(self):
        """Returns a string which can be used to reproduce an instance of Matrix."""
        return "Matrix([" + "],\n       [".\
        join(", ".join((str(j) for j in self.components[i])) for i in range(4)) + "])"

    def __getitem__(self, ij):
        """Returns the element at row ij[0], column ij[1]. Can be accessed through
           Matrix_instance[i, j] for row i, column j."""
        return self.components[ij[0]][ij[1]]

    def __setitem__(self, ij, val):
        """Sets the element at row ij[0], column ij[1] to the value of 'val'."""
        self.components[ij[0]][ij[1]] = val

    def __add__(self, other):
        """Returns the sum of two instances of Matrix, self and 'other'."""
        result = self.__class__([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
        for i in range(4):
            for j in range(4):
                result[i, j] = self[i, j] + other[i, j]
        return result

    def __iadd__(self, other):
        """Augmented assignment for addition of two instances of Matrix."""
        self = self + other
        return self

    def __sub__(self, other):
        """Returns the result of the subtraction of 'other' from an instance
           Matrix, self."""
        result = self.__class__([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
        for i in range(4):
            for j in range(4):
                result[i, j] = self[i, j] - other[i, j]
        return result

    def __isub__(self, other):
        """Augmented assignment for subtraction of 'other' from an instance of Matrix."""
        self = self - other
        return self

    def __pos__(self):
        """Implementation for the unary '+' operator. Returns the instance of
           Matrix with no modification."""
        return self

    def __neg__(self):
        """Implementation of the unary '-' operator. Returns the instance of
           Matrix with all element signs flipped."""
        return self.__class__([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]) - self

    def __invert__(self):
        """Returns conjugate transpose of Matrix."""
        result = self.__class__([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
        for i in range(4):
            for j in range(4):
                result[i, j] = self[j, i].conjugate()
        return result

    def __mul__(self, other):
        """Returns the result of an instance of Matrix being multiplied by 'other'.
           'other' may be a scalar, Vector or Matrix.

           -If it is a scalar, each element of the Matrix will be multiplied by 'other'.
           -If it is a Vector, the product of 'other' and self will be returned,
           as a Vector. This applies for the order: Matrix*Vector.
           -If it is a Matrix, the product of 'other' and self will
           be returned, as a Matrix or an inherited class, in accordance with the
           rules of matrix algebra."""

        if isinstance(other, Matrix): # M*M.
            result = self.__class__([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        result[i, j] += self[i, k]*other[k, j]
            return result

        elif isinstance(other, Vector): # M*v.
            result = other.__class__(0, 0, 0, 0)
            for i in range(4):
                for j in range(4):
                    result[i] += (self[i, j]*other[j])
            return result

        else: # Assume other is a scalar numeric type, M*s.
            result = self.__class__([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
            for i in range(4):
                for j in range(4):
                    result[i, j] = self[i, j]*other
            return result

    def __rmul__(self, other): # Don't need to account for M*M here.
        """Returns the result of multiplying an instance of Matrix with 'other'. This is
        done in the same way as __mul__ if 'other' is a scalar, however if it is a Vector,
        then the Vector will be transposed and multiplied with the instance of Matrix.
        This applies for the order: Vector*matrix."""

        if isinstance(other, Vector): # v*M.
            result = other.__class__(0, 0, 0, 0)
            for i in range(4):
                for j in range(4):
                    result[i] += other[j]*self.components[j][i]
            return result

        else: # Assume s*M.
            return self*other

    def __imul__(self, other):
        """Augmented assignment for multiplication of Matrix with 'other'. Follows same
           rules as __mul__."""
        self = self*other
        return self

    def __truediv__(self, other):
        """Returns the result of dividing an instance of Matrix by a scalar, 'other'.
           Each element of the Matrix will be divided by the scalar."""
        result = self.__class__([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
        for i in range(4):
            for j in range(4):
                result[i, j] = self[i, j]/other # Assume only input will be scalar.
        return result

    def __idiv__(self, other):
        """Augmented assignment for division of a Matrix by a scalar, 'other'.
           Each element of the Matrix will be divided by the scalar."""
        self = self/other
        return self

    def __abs__(self):
        """Returns the Frobenius norm of a Matrix instance."""
        result = 0
        for i in range(4):
            for j in range(4):
                result += self[i, j]*self[i, j].conjugate()
        return result.real**0.5

    def __pow__(self, power):
        """Returns the result of a Matrix instance raised to a given power 'power'.
           If that power is 0, then the 4x4 identity matrix is returned. Otherwise,
           the Matrix multiplied by itself that integer number of times is returned."""

        if power == 0:
            return self.__class__([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
        result = self
        for _ in range(int(power-1)): # Assumes integer power (will cast to int).
            result = result*self
        return result

    def __ipow__(self, power):
        """Augmented assignment for exponentiation of Matrix instances,
           follows __pow__ rules."""
        self = self**power
        return self



class FourVector(Vector):
    """A Four Vector class which inherits from Vector, maintaining the same arithmetic
       methods as the standard Vector class. Problem 3.

       The same constructor is used. Initialise by presenting the values of each
       of the 4 elements as separate arguments."""

    def __repr__(self):
        """Returns a string which can be used to reproduce an instance of FourVector."""
        return "FourVector(" + ", ".join(str(i) for i in self.components) + ")"

    def __abs__(self):
        """Returns the norm of the FourVector."""
        return self*Matrix([1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1])*self

    def __invert__(self):
        """Defines the unary '~' operator. Returns a contravariant tensor if the
           FourVector is covariant, or a covariant tensor if the FourVector
           is contravariant."""
        return Matrix([1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1])*self


from sys import float_info

class BoostMatrix(Matrix):
    """A BoostMatrix class which inherits from Matrix, and maintains the same
       arithmetic methods."""

    def __init__(self, p, A=None, B=None, C=None):
        """The constructor for the BoostMatrix class. The BoostMatrix should be
           initialised by passing a single FourVector as an argument to the constructor.
           This will automatically calculate the correct BoostMatrix elements for that
           FourVector's reference frame.

           Alternatively, it could be initialised by passing 4 individual iterables
           of length 4 to the constructor, as separate arguments, in order to set the
           elements of the matrix directly. Each iterable should act as a row.
           If fewer than 4 arguments are passed, the 'p' argument will be taken as
           a FourVector and used as described above.
           
           If the argument(s) given would result in the BoostMatrix gamma value exceeding
           float range, an exception will be raised."""

        if A and B and C:
            self.components = [list(p), list(A), list(B), list(C)]
            # This is a solution to get the inherited arithmetic methods and new repr
            # to work, since they access the elements directly, through the constructor.

            # The BoostMatrix can still be initialised with the 'p' FourVector alone.

        else:
            bx, by, bz = p[1]/p[0], p[2]/p[0], p[3]/p[0] # beta = p/E = x/ct
            beta = ((bx**2 + by**2 + bz**2)**0.5)
            gamma = 1/(1 - beta**2)**0.5
            alpha = (gamma**2)/(1 + gamma)

            self.components = [[gamma, -gamma*bx, -gamma*by, -gamma*bz],\
                               [-gamma*bx, 1+alpha*(bx**2), alpha*bx*by, alpha*bx*bz],\
                               [-gamma*by, alpha*by*bz, 1+alpha*(by**2), alpha*by*bz],\
                               [-gamma*bz, alpha*bz*bx, alpha*bz*by, 1+alpha*(bz**2)]]

        if abs(self.components[0][0]) >= float_info.max: # Check if 'inf' or 'float max'.
            raise Exception("Boost too large, gamma exceeded float range.")

    def __repr__(self):
        """Returns a string which can be used to reproduce an instance of BoostMatrix."""
        return "BoostMatrix([" + "],\n            [".\
        join(", ".join((str(j) for j in self.components[i])) for i in range(4)) + "])"
