{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleInstances, ConstraintKinds #-}

module Optimierung where 

import GHC.Exts
import qualified Control.Foldl as L

class Optimierung d p where 
    data Akkum d p

    regel :: d p -> Akkum d p -> p -> Akkum d p 
    start :: p -> Akkum d p
    aus   :: Akkum d p -> p

    --1. Regel: aus . start = id

class Opt d where
    data Ak d :: * -> *
    data Con d :: * -> * 
    type Cons d p :: Constraint 
    
    regel' :: Cons d p => d p -> Ak d p -> p -> Ak d p

--Optimierungsmethoden:
newtype L2 p = L2 {unL2 :: (p, p)}
newtype SGD p = SGD {unSGD :: p}
newtype Minibatch d p = Minibatch {unMinibatch :: (Int, d p)}

instance Opt L2 where
    data Ak L2 p = OpL2' {unOpL2' :: p}
    
    data Con L2 p =  K {unK :: p}
    type Cons L2 p = Fractional p

    regel' = undefined
    
instance Fractional p => Optimierung L2 p where
    newtype Akkum L2 p = OpL2 {unOpL2 :: p}
    
    regel (L2 (r, l)) (OpL2 p) g = OpL2 $ (1 - r*l) * p - r * g
    start = OpL2
    aus   = unOpL2

instance Fractional p => Optimierung SGD p where
    newtype Akkum SGD p = OpSGD {unOpSGD :: p}

    regel (SGD r) (OpSGD p) g = OpSGD $ p - r * g
    start = OpSGD
    aus   = unOpSGD
 
instance (Optimierung d p, Fractional p) => Optimierung (Minibatch d) p where
    newtype Akkum (Minibatch d) p = OpMinibatch {unOpMinibatch :: (Int, p, Akkum d p)}

    regel (Minibatch (n, d)) (OpMinibatch (m, q, p)) g 
        | m == n    = OpMinibatch (0, 0, regel d p ((q + g) / fromIntegral n))
        | otherwise = OpMinibatch (m+1, q + g, p)
    start p = OpMinibatch (0, 0, start p)
    aus (OpMinibatch (_, _, p)) = aus p
