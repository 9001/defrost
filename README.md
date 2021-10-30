# defrost
* split broken icecast recordings into separate mp3s
* supports windows, macos, linux, freebsd
* MIT-Licensed, 2019-06-01, ed @ irc.rizon.net

# features
* **silence detection** -- finds the best part to split at
* **mp3 parser** -- clean splits at frame boundaries
* **icy-meta parser** -- extract artist/title from inband metadata
  * dynamic recalibration to recover from server glitches
* **charset detection** -- UTF8, latin-1, shift-jis and friends, all good
* **ID3 tags** -- uses mutagen to write aritst/title into the files
* **timestamping** -- the date/time that the song was played live
* support for other input formats:
  * loopstream recording
