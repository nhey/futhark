-- | A usage-table is sort of a bottom-up symbol table, describing how
-- (and if) a variable is used.
module Futhark.EnablingOpts.UsageTable
  ( UsageTable
  , empty
  , contains
  , without
  , lookup
  , used
  , usedAsPredicate
  , usages
  , predicateUsage
  , Usages
  , isPredicate
  )
  where

import Prelude hiding (lookup, any, foldl)

import Data.Foldable
import qualified Data.HashMap.Lazy as HM
import qualified Data.HashSet as HS
import qualified Data.Set as S

import Futhark.InternalRep

type UsageTable = HM.HashMap VName Usages

empty :: UsageTable
empty = HM.empty

contains :: UsageTable -> [VName] -> Bool
contains table = any (`HM.member` table)

without :: UsageTable -> [VName] -> UsageTable
without = foldl (flip HM.delete)

lookup :: VName -> UsageTable -> Maybe Usages
lookup = HM.lookup

lookupPred :: (Usages -> Bool) -> VName -> UsageTable -> Bool
lookupPred f name = maybe False f . lookup name

used :: VName -> UsageTable -> Bool
used = lookupPred $ const True

usedAsPredicate :: VName -> UsageTable -> Bool
usedAsPredicate = lookupPred isPredicate

usages :: HS.HashSet VName -> UsageTable
usages names = HM.fromList [ (name, S.empty) | name <- HS.toList names ]

predicateUsage :: VName -> UsageTable
predicateUsage name = HM.singleton name $ S.singleton Predicate

type Usages = S.Set Usage

data Usage = Predicate
             deriving (Eq, Ord, Show)

isPredicate :: Usages -> Bool
isPredicate = S.member Predicate
