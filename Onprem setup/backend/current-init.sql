-- Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Creating the users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    login_count INTEGER DEFAULT 0,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Creating the roles table
CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Creating the permissions table
CREATE TABLE permissions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Creating the user_roles table to link users to roles
CREATE TABLE user_roles (
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, role_id)
);

-- Creating the role_permissions table to link roles to permissions
CREATE TABLE role_permissions (
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    permission_id INTEGER REFERENCES permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

-- Creating the audit_logs table
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Creating the reports table with a unique constraint on (study_id, created_by)
CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    study_id TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    study_type TEXT NOT NULL,
    study_date DATE NOT NULL, -- Changed to DATE for better date handling
    study_description TEXT NOT NULL,
    findings TEXT NOT NULL,
    impression TEXT NOT NULL,
    recommendations TEXT,
    radiologist_name TEXT NOT NULL,
    report_date TEXT NOT NULL,
    created_by VARCHAR(50) NOT NULL REFERENCES users(username) ON DELETE CASCADE, -- Added foreign key constraint
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'completed')),
    notes TEXT,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    patient_age VARCHAR(10),
    patient_age_years INTEGER, -- Added for storing parsed age in years
    patient_sex VARCHAR(1),
    ai_findings JSONB, -- Added for AI results
    CONSTRAINT unique_study_createdby UNIQUE (study_id, created_by)
);

-- Creating the study_views table
CREATE TABLE study_views (
    id SERIAL PRIMARY KEY,
    study_id TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    viewer_type VARCHAR(20),
    view_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Creating the report_sessions table with a unique constraint on (study_id, user_id)
CREATE TABLE report_sessions (
    id SERIAL PRIMARY KEY,
    study_id TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    completed BOOLEAN DEFAULT FALSE,
    CONSTRAINT unique_study_user_session UNIQUE (study_id, user_id)
);

-- Creating the ai_analysis_sessions table
CREATE TABLE ai_analysis_sessions (
    id SERIAL PRIMARY KEY,
    study_id TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    analysis_type VARCHAR(50) DEFAULT 'default',
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed')),
    result JSONB,
    CONSTRAINT unique_study_user_analysis UNIQUE (study_id, user_id)
);

-- Creating the study_assignments table
CREATE TABLE study_assignments (
    id SERIAL PRIMARY KEY,
    study_id TEXT NOT NULL,
    assigned_to INTEGER REFERENCES users(id) ON DELETE SET NULL,
    assigned_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    assigned_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completion VARCHAR(10) DEFAULT '0%' CHECK (completion IN ('0%', '25%', '50%', '75%', '100%')),
    CONSTRAINT unique_study_assignment UNIQUE (study_id, assigned_to)
);

-- Creating the emails table (with is_read column)
CREATE TABLE emails (
    id SERIAL PRIMARY KEY,
    sender_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    receiver_id INTEGER REFERENCES users(id) ON DELETE CASCADE, -- NULL for external emails
    receiver_email VARCHAR(100), -- Used for external emails
    subject VARCHAR(255) NOT NULL,
    body TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'sent' CHECK (status IN ('draft', 'sent', 'failed')),
    is_external BOOLEAN DEFAULT FALSE,
    is_read BOOLEAN DEFAULT FALSE,
    CONSTRAINT check_receiver CHECK (
        (is_external = TRUE AND receiver_email IS NOT NULL AND receiver_id IS NULL) OR
        (is_external = FALSE AND receiver_id IS NOT NULL AND receiver_email IS NULL)
    )
);

-- Adding indexes for performance
CREATE INDEX idx_reports_study_createdby ON reports(study_id, created_by);
CREATE INDEX idx_report_sessions_study_user ON report_sessions(study_id, user_id);
CREATE INDEX idx_reports_study_date ON reports(study_date);
CREATE INDEX idx_reports_study_description ON reports(study_description);
CREATE INDEX idx_reports_patient_age ON reports(patient_age);
CREATE INDEX idx_reports_patient_sex ON reports(patient_sex);
CREATE INDEX idx_reports_created_by ON reports(created_by);
CREATE INDEX idx_reports_study_type ON reports(study_type);
CREATE INDEX idx_reports_status ON reports(status);
CREATE INDEX idx_reports_patient_age_years ON reports(patient_age_years);
CREATE INDEX idx_ai_analysis_sessions_study_user ON ai_analysis_sessions(study_id, user_id);
CREATE INDEX idx_study_assignments_study_id ON study_assignments(study_id);
CREATE INDEX idx_study_assignments_assigned_to ON study_assignments(assigned_to);
CREATE INDEX idx_emails_sender_id ON emails(sender_id);
CREATE INDEX idx_emails_receiver_id ON emails(receiver_id);
CREATE INDEX idx_emails_timestamp ON emails(timestamp);
CREATE INDEX idx_emails_is_external ON emails(is_external);
CREATE INDEX idx_emails_is_read ON emails(is_read);

-- Inserting permissions
INSERT INTO permissions (name) VALUES
('search_studies'),
('assign_studies'),
('view_reports'),
('manage_users'),
('export_audit_logs'),
('create_report'),
('upload_pdf'),
('view_system'),
('view_mails'),
('view_recent_studies'),
('view_assigned_studies'),
('share_studies'),
('use_ai'),
('view_statistics'),
('send_emails'),
('view_studies');

-- Inserting roles
INSERT INTO roles (name, description) VALUES
('Admin', 'Full access to all features including user management'),
('Doctor', 'Access to search and assign studies');

-- Inserting role_permissions
-- Admin role gets all permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT 1, id FROM permissions;

-- Doctor role gets specific permissions (added view_mails)
INSERT INTO role_permissions (role_id, permission_id)
VALUES
(2, (SELECT id FROM permissions WHERE name = 'search_studies')),
(2, (SELECT id FROM permissions WHERE name = 'assign_studies')),
(2, (SELECT id FROM permissions WHERE name = 'view_reports')),
(2, (SELECT id FROM permissions WHERE name = 'create_report')),
(2, (SELECT id FROM permissions WHERE name = 'upload_pdf')),
(2, (SELECT id FROM permissions WHERE name = 'view_system')),
(2, (SELECT id FROM permissions WHERE name = 'view_mails')), -- Added for email fetching
(2, (SELECT id FROM permissions WHERE name = 'view_recent_studies')),
(2, (SELECT id FROM permissions WHERE name = 'view_assigned_studies')),
(2, (SELECT id FROM permissions WHERE name = 'share_studies')),
(2, (SELECT id FROM permissions WHERE name = 'send_emails')),
(2, (SELECT id FROM permissions WHERE name = 'view_studies'));

-- Inserting users with pgcrypto bcrypt hashes
INSERT INTO users (username, email, password_hash) VALUES
('admin', 'admin@example.com', crypt('admin', gen_salt('bf', 12))),
('doctor', 'doctor@example.com', crypt('doctor', gen_salt('bf', 12)));

-- Inserting user_roles
INSERT INTO user_roles (user_id, role_id) VALUES
(1, 1), -- admin -> Admin role
(2, 2); -- doctor -> Doctor role

-- Inserting sample emails for testing
-- Email 1: Sent email from admin to doctor (inbox for doctor, sent for admin)
INSERT INTO emails (sender_id, receiver_id, subject, body, timestamp, status, is_external, is_read)
VALUES
((SELECT id FROM users WHERE username = 'admin'), 
 (SELECT id FROM users WHERE username = 'doctor'), 
 'Welcome to the System', 
 'Hi Doctor, welcome to the Medical Imaging Dashboard!', 
 CURRENT_TIMESTAMP - INTERVAL '2 days', 
 'sent', 
 FALSE, 
 FALSE);

-- Email 2: Draft email by doctor (outbox for doctor)
INSERT INTO emails (sender_id, receiver_id, subject, body, timestamp, status, is_external, is_read)
VALUES
((SELECT id FROM users WHERE username = 'doctor'), 
 (SELECT id FROM users WHERE username = 'admin'), 
 'Draft Report Review', 
 'Hi Admin, please review this draft report.', 
 CURRENT_TIMESTAMP - INTERVAL '1 day', 
 'draft', 
 FALSE, 
 FALSE);

-- Email 3: Sent external email from admin (sent for admin)
INSERT INTO emails (sender_id, receiver_email, subject, body, timestamp, status, is_external, is_read)
VALUES
((SELECT id FROM users WHERE username = 'admin'), 
 'external@example.com', 
 'External Test Email', 
 'This is a test email sent externally.', 
 CURRENT_TIMESTAMP - INTERVAL '3 days', 
 'sent', 
 TRUE, 
 FALSE);

-- Email 4: Sent email from doctor to admin (inbox for admin, sent for doctor, marked as read)
INSERT INTO emails (sender_id, receiver_id, subject, body, timestamp, status, is_external, is_read)
VALUES
((SELECT id FROM users WHERE username = 'doctor'), 
 (SELECT id FROM users WHERE username = 'admin'), 
 'Follow-Up on Study', 
 'Hi Admin, can we discuss the recent study?', 
 CURRENT_TIMESTAMP - INTERVAL '4 hours', 
 'sent', 
 FALSE, 
 TRUE);

-- Cleanup script for inactive draft reports
DO $$
DECLARE
    v_current_timestamp TIMESTAMP WITH TIME ZONE := '2025-06-07 15:32:00+01:00'; -- 03:32 PM WAT
    v_inactivity_threshold INTERVAL := INTERVAL '4320 minutes';
    v_report RECORD;
    v_admin_user_id INTEGER := (SELECT id FROM users WHERE username = 'admin');
BEGIN
    -- Step 1: Identify draft reports and sessions inactive for over 10 minutes
    FOR v_report IN (
        SELECT r.id AS report_id, rs.id AS session_id, r.study_id, r.created_by
        FROM reports r
        LEFT JOIN report_sessions rs ON r.study_id = rs.study_id
        WHERE r.status = 'draft'
        AND (rs.end_time IS NULL OR rs.completed = FALSE)
        AND (v_current_timestamp - COALESCE(rs.start_time, r.start_time, r.created_at)) > v_inactivity_threshold
    ) LOOP
        -- Step 2: Update report_sessions to mark as closed
        UPDATE report_sessions
        SET end_time = v_current_timestamp,
            completed = TRUE
        WHERE id = v_report.session_id;

        -- Step 3: Delete the draft report
        DELETE FROM reports
        WHERE id = v_report.report_id;

        -- Step 4: Log the cleanup action in audit_logs
        INSERT INTO audit_logs (user_id, action, details, timestamp)
        VALUES (
            v_admin_user_id,
            'cleanup_inactive_report',
            jsonb_build_object(
                'study_id', v_report.study_id,
                'created_by', v_report.created_by,
                'reason', 'Inactive for over 10 minutes'
            ),
            v_current_timestamp
        );
    END LOOP;

    -- Step 5: Log completion of cleanup
    INSERT INTO audit_logs (user_id, action, details, timestamp)
    VALUES (
        v_admin_user_id,
        'cleanup_completed',
        jsonb_build_object('message', 'Cleanup of inactive draft reports completed'),
        v_current_timestamp
    );
END $$;