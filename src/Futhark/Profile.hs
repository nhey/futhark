-- | Profiling information emitted by a running Futhark program.
module Futhark.Profile
  ( ProfilingEvent (..),
    ProfilingReport (..),
    profilingInfoFromText,
  )
where

import Data.Aeson qualified as JSON
import Data.Aeson.Key qualified as JSON
import Data.Aeson.KeyMap qualified as JSON
import Data.Bifunctor
import Data.ByteString.Builder (toLazyByteString)
import Data.Map qualified as M
import Data.Text qualified as T
import Data.Text.Encoding (encodeUtf8Builder)

-- | A thing that has occurred during execution.
data ProfilingEvent = ProfilingEvent
  { -- | Short, single line.
    eventName :: T.Text,
    -- | In microseconds.
    eventDuration :: Double,
    -- | Long, may be multiple lines.
    eventDescription :: T.Text
  }
  deriving (Eq, Ord, Show)

instance JSON.ToJSON ProfilingEvent where
  toJSON (ProfilingEvent name duration description) =
    JSON.object
      [ ("name", JSON.toJSON name),
        ("duration", JSON.toJSON duration),
        ("description", JSON.toJSON description)
      ]

instance JSON.FromJSON ProfilingEvent where
  parseJSON = JSON.withObject "event" $ \o ->
    ProfilingEvent
      <$> o JSON..: "name"
      <*> o JSON..: "duration"
      <*> o JSON..: "description"

data ProfilingReport = ProfilingReport
  { profilingEvents :: [ProfilingEvent],
    -- | Mapping memory spaces to bytes.
    profilingMemory :: M.Map T.Text Integer
  }
  deriving (Eq, Ord, Show)

instance JSON.ToJSON ProfilingReport where
  toJSON (ProfilingReport events memory) =
    JSON.object
      [ ("events", JSON.toJSON events),
        ("memory", JSON.object $ map (bimap JSON.fromText JSON.toJSON) $ M.toList memory)
      ]

instance JSON.FromJSON ProfilingReport where
  parseJSON = JSON.withObject "profiling-info" $ \o ->
    ProfilingReport
      <$> o JSON..: "events"
      <*> (JSON.toMapText <$> o JSON..: "memory")

profilingInfoFromText :: T.Text -> Maybe ProfilingReport
profilingInfoFromText = JSON.decode . toLazyByteString . encodeUtf8Builder
