-- =============================================================================
-- 009_email_verification.sql
-- Adds email verification and OTP tracking columns to users table
-- =============================================================================

ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_otp_hash TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_expires_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_otp_sent_at TIMESTAMPTZ;

COMMENT ON COLUMN users.verification_otp_hash IS 'Salted PBKDF2-HMAC-SHA256 hash of the 6-digit email verification OTP (never stored plaintext)';
COMMENT ON COLUMN users.verification_expires_at IS 'Expiration timestamp for the current verification OTP';
COMMENT ON COLUMN users.verification_attempts IS 'Number of consecutive failed OTP verification attempts';
COMMENT ON COLUMN users.last_otp_sent_at IS 'Timestamp when the last verification OTP was dispatched (used for 60s rate limit / cooldown)';
