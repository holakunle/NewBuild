from fastapi import FastAPI, HTTPException, Depends, Body, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl, ValidationError
from PIL import Image
from typing import List, Optional, Dict, Any
import asyncpg
import jwt
import os
from datetime import datetime, timedelta, date
import json
import logging
import requests
import uuid
import pydicom
import torch
import torch.nn as nn
import numpy as np
from io import BytesIO
import cv2
from torchvision import transforms, models
from dotenv import load_dotenv
from mailersend import emails

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variable to hold the connection pool

# Global variable to hold the connection pool
db_pool = None

# Startup event to initialize the connection pool
@app.on_event("startup")
async def startup_event():
    global db_pool
    logger.info("Initializing database connection pool")
    db_pool = await asyncpg.create_pool(
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'password'),
        database=os.getenv('DB_NAME', 'mydb'),
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        min_size=5,  # Minimum number of connections to keep in the pool
        max_size=50  # Increased from 20 to 50 for better concurrency
    )
    logger.info("Database connection pool initialized")

# Shutdown event to close the connection pool
@app.on_event("shutdown")
async def shutdown_event():
    global db_pool
    logger.info("Closing database connection pool")
    await db_pool.close()
    logger.info("Database connection pool closed")

# Dependency to provide the pool to endpoints
async def get_db():
    if db_pool is None:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    return db_pool

# Function to load Orthanc settings from environment variables
def get_orthanc_settings():
    default_url = "http://orthanc:8042"
    orthanc_url = os.getenv("ORTHANC_URL", default_url).rstrip("/")
    if orthanc_url.endswith("/orthanc"):
        logger.info(f"Substituting ORTHANC_URL ({orthanc_url}) with internal Docker URL ({default_url}) for backend requests")
        internal_url = default_url
    else:
        internal_url = orthanc_url
    if not internal_url.startswith("http://") and not internal_url.startswith("https://"):
        internal_url = f"http://{internal_url}"
    return {
        "url": internal_url,
        "external_url": orthanc_url,
        "username": os.getenv("ORTHANC_USERNAME", "admin"),
        "password": os.getenv("ORTHANC_PASSWORD", "password")
    }

# Function to resolve StudyInstanceUID to internal Orthanc study ID
async def resolve_study_id(study_id: str, current_user: dict) -> str:
    logger.info(f"Resolving study ID {study_id} for user {current_user['username']}")
    settings = get_orthanc_settings()
    try:
        # First, try to fetch the study assuming study_id is an internal Orthanc ID
        study_response = requests.get(
            f"{settings['url']}/studies/{study_id}",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        if study_response.status_code == 200:
            logger.info(f"Study ID {study_id} is an internal Orthanc ID")
            return study_id
        else:
            # Assume study_id is a StudyInstanceUID and resolve to internal ID
            logger.info(f"Study ID {study_id} not found as internal ID, attempting to resolve as StudyInstanceUID")
            find_response = requests.post(
                f"{settings['url']}/tools/find",
                json={
                    "Level": "Study",
                    "Query": {
                        "StudyInstanceUID": study_id
                    }
                },
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            find_response.raise_for_status()
            study_ids = find_response.json()
            logger.info(f"Found {len(study_ids)} studies for StudyInstanceUID {study_id}")
            if not study_ids:
                logger.error(f"No study found for StudyInstanceUID {study_id}")
                raise HTTPException(status_code=404, detail=f"Study with StudyInstanceUID {study_id} not found in Orthanc")
            if len(study_ids) > 1:
                logger.warning(f"Multiple studies found for StudyInstanceUID {study_id}: {study_ids}")
            internal_study_id = study_ids[0]
            logger.info(f"Resolved StudyInstanceUID {study_id} to internal study ID {internal_study_id}")
            return internal_study_id
    except requests.RequestException as e:
        logger.error(f"Error resolving study ID {study_id}: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to resolve study ID from Orthanc")

# Pydantic models
# Pydantic models
class OrthancConfig(BaseModel):
    orthanc_url: HttpUrl

class HospitalConfig(BaseModel):
    hospital_name: str
    hospital_address: str
    hospital_phone: str
    hospital_logo: Optional[str] = None

class StudyQuery(BaseModel):
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    study_date_from: Optional[str] = None
    study_date_to: Optional[str] = None
    modality: Optional[str] = None
    study_description: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None
    study_status: Optional[str] = None
    labels: Optional[List[str]] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateUserRequest(BaseModel):
    username: str
    email: Optional[str] = None
    password: str

class CreateRoleRequest(BaseModel):
    name: str
    description: str
    permission_ids: List[int]

class UpdateUserRolesRequest(BaseModel):
    role_ids: List[int]

class Report(BaseModel):
    id: Optional[int] = None
    studyId: str
    patientName: str
    patientID: str
    studyType: str
    studyDate: str
    studyDescription: str
    findings: str
    impression: str
    recommendations: Optional[str] = None
    radiologistName: str
    reportDate: str
    createdBy: Optional[str] = None
    createdAt: Optional[str] = None
    status: Optional[str] = "draft"
    notes: Optional[str] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    ai_findings: Optional[Dict] = None  # Added for AI findings

class StudyView(BaseModel):
    study_id: str
    viewer_type: str

class ReportSession(BaseModel):
    study_id: str

class ReassignStudy(BaseModel):
    study_id: str
    new_doctor: str

class ReportComplete(BaseModel):
    study_id: str

class EmailRequest(BaseModel):
    recipient: str
    recipient_type: str
    subject: str
    body: str

class EmailActionRequest(BaseModel):
    email_ids: List[int]

class AIRequest(BaseModel):
    study_id: str
    analysis_type: Optional[str] = "default"

class AIResult(BaseModel):
    study_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    specific_findings: Optional[str] = None
    recommendations: Optional[str] = None
    supporting_metrics: Optional[Dict[str, Any]] = None
    severity: Optional[str] = None

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: asyncpg.Pool = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, os.getenv('JWT_SECRET', 'your-secret-key'), algorithms=['HS256'])
        username = payload.get('sub')
        logger.info(f"JWT payload: {payload}")
        if not username:
            logger.error('Invalid token: No username in payload')
            raise HTTPException(status_code=401, detail="Invalid token")
        async with db.acquire() as conn:
            user = await conn.fetchrow('SELECT id, username FROM users WHERE username = $1', username)
            if not user:
                logger.error(f"User not found: {username}")
                raise HTTPException(status_code=401, detail="User not found")
            logger.info(f"Resolved user: id={user['id']}, username={user['username']}")
        return {"id": user["id"], "username": username}
    except jwt.PyJWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

async def check_permission(user_id: int, permission: str, db: asyncpg.Pool):
    async with db.acquire() as conn:
        has_permission = await conn.fetchval('''
            SELECT EXISTS (
                SELECT 1
                FROM user_roles ur
                JOIN role_permissions rp ON ur.role_id = rp.role_id
                JOIN permissions p ON rp.permission_id = p.id
                WHERE ur.user_id = $1 AND p.name = $2
            )
        ''', user_id, permission)
        if not has_permission:
            logger.error(f"Permission denied for user ID {user_id}: {permission}")
            raise HTTPException(status_code=403, detail=f"Permission denied: {permission} required")
        logger.info(f"Permission {permission} granted for user ID {user_id}")

# Existing endpoints
# Existing endpoints
@app.get("/api/users")
async def get_users(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching users by {current_user['username']}")
    await check_permission(current_user['id'], 'manage_users', db)
    async with db.acquire() as conn:
        users = await conn.fetch('''
            SELECT u.id, u.username, u.email, u.is_active, u.login_count, u.last_login, u.created_at,
                   COALESCE(json_agg(row_to_json(r)) FILTER (WHERE r.id IS NOT NULL), '[]') as roles,
                   COALESCE((
                       SELECT json_agg(json_build_object('name', p.name))
                       FROM permissions p
                       JOIN role_permissions rp ON p.id = rp.permission_id
                       JOIN user_roles ur2 ON rp.role_id = ur2.role_id
                       WHERE ur2.user_id = u.id
                       AND p.name IS NOT NULL
                   ), '[]') as permissions
            FROM users u
            LEFT JOIN user_roles ur ON u.id = ur.user_id
            LEFT JOIN roles r ON ur.role_id = r.id
            GROUP BY u.id
            HAVING u.username IS NOT NULL
        ''')
        logger.info(f"Retrieved {len(users)} users")
        return [{
            'id': u['id'],
            'username': u['username'],
            'email': u['email'],
            'is_active': u['is_active'],
            'login_count': u['login_count'],
            'last_login': u['last_login'].isoformat() if u['last_login'] else None,
            'created_at': u['created_at'].isoformat(),
            'roles': json.loads(u['roles']) if isinstance(u['roles'], str) else u['roles'],
            'permissions': json.loads(u['permissions']) if isinstance(u['permissions'], str) else u['permissions']
        } for u in users]
    
@app.get("/api/internal-users")
async def get_internal_users(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching internal users for {current_user['username']}")
    await check_permission(current_user['id'], 'send_emails', db)
    async with db.acquire() as conn:
        users = await conn.fetch('''
            SELECT username
            FROM users
            WHERE username != $1 AND is_active = TRUE
            ORDER BY username
        ''', current_user['username'])
        logger.info(f"Retrieved {len(users)} internal users for {current_user['username']}")
        return [user['username'] for user in users]

@app.get("/api/current-user")
async def get_current_user_info(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching current user info for {current_user['username']}")
    async with db.acquire() as conn:
        user = await conn.fetchrow('''
            SELECT id, username, email, is_active, login_count, last_login, created_at,
                   COALESCE((
                       SELECT json_agg(json_build_object('name', p.name))
                       FROM permissions p
                       JOIN role_permissions rp ON p.id = rp.permission_id
                       JOIN user_roles ur ON rp.role_id = ur.role_id
                       WHERE ur.user_id = u.id
                       AND p.name IS NOT NULL
                   ), '[]') as permissions
            FROM users u
            WHERE username = $1
        ''', current_user['username'])
        if not user:
            logger.error(f"User not found: {current_user['username']}")
            raise HTTPException(status_code=404, detail="User not found")
        return {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'is_active': user['is_active'],
            'login_count': user['login_count'],
            'last_login': user['last_login'].isoformat() if user['last_login'] else None,
            'created_at': user['created_at'].isoformat(),
            'permissions': json.loads(user['permissions']) if isinstance(user['permissions'], str) else user['permissions']
        }

@app.post("/api/auth/login")
async def login(request: LoginRequest, db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Login attempt for user: {request.username}")
    async with db.acquire() as conn:
        user = await conn.fetchrow('''
            SELECT id, username, email, is_active, login_count, last_login, created_at
            FROM users 
            WHERE username = $1 AND password_hash = crypt($2, password_hash)
        ''', request.username, request.password)
        if user:
            token = jwt.encode({
                'sub': request.username,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, os.getenv('JWT_SECRET', 'your-secret-key'), algorithm='HS256')
            
            await conn.execute('''
                UPDATE users 
                SET login_count = login_count + 1, 
                    last_login = $1 
                WHERE username = $2
            ''', datetime.utcnow(), request.username)
            
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', user['id'], 'login', json.dumps({'username': request.username}), datetime.utcnow())
            
            logger.info(f"Login successful for user: {request.username}")
            return {'token': token}
        logger.error(f"Invalid credentials for user: {request.username}")
        raise HTTPException(status_code=401, detail='Invalid credentials')

@app.post("/api/users")
async def create_user(request: CreateUserRequest, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Creating user: {request.username} by {current_user['username']}")
    await check_permission(current_user['id'], 'manage_users', db)
    async with db.acquire() as conn:
        try:
            user_id = await conn.fetchval('''
                INSERT INTO users (username, email, password_hash)
                VALUES ($1, $2, crypt($3, gen_salt('bf', 12)))
                RETURNING id
            ''', request.username, request.email, request.password)
            
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', user_id, 'create_user', json.dumps({'username': request.username}), datetime.utcnow())
            
            logger.info(f"User created: {request.username}")
            return {'message': 'User created successfully'}
        except asyncpg.UniqueViolationError:
            logger.error(f"Username already exists: {request.username}")
            raise HTTPException(status_code=400, detail='Username already exists')
        except Exception as e:
            logger.error(f"Error creating user {request.username}: {str(e)}")
            raise HTTPException(status_code=500, detail='Internal server error')

@app.get("/api/permissions")
async def get_permissions(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching permissions by {current_user['username']}")
    await check_permission(current_user['id'], 'manage_users', db)
    async with db.acquire() as conn:
        permissions = await conn.fetch('''
            SELECT id, name
            FROM permissions
        ''')
        logger.info(f"Retrieved {len(permissions)} permissions")
        return [{'id': p['id'], 'name': p['name']} for p in permissions]

@app.get("/api/roles")
async def get_roles(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching roles by {current_user['username']}")
    await check_permission(current_user['id'], 'manage_users', db)
    async with db.acquire() as conn:
        roles = await conn.fetch('''
            SELECT r.id, r.name, r.description, r.created_at,
                   COALESCE((
                       SELECT json_agg(json_build_object('id', p.id, 'name', p.name))
                       FROM permissions p
                       JOIN role_permissions rp ON p.id = rp.permission_id
                       WHERE rp.role_id = r.id
                   ), '[]') as permissions
            FROM roles r
            GROUP BY r.id
        ''')
        logger.info(f"Retrieved {len(roles)} roles")
        return [{
            'id': r['id'],
            'name': r['name'],
            'description': r['description'],
            'created_at': r['created_at'].isoformat(),
            'permissions': json.loads(r['permissions']) if isinstance(r['permissions'], str) else r['permissions']
        } for r in roles]

@app.post("/api/roles")
async def create_role(request: CreateRoleRequest, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Creating role: {request.name} by {current_user['username']}")
    await check_permission(current_user['id'], 'manage_users', db)
    async with db.acquire() as conn:
        async with conn.transaction():
            try:
                role_id = await conn.fetchval('''
                    INSERT INTO roles (name, description, created_at)
                    VALUES ($1, $2, $3)
                    RETURNING id
                ''', request.name, request.description, datetime.utcnow())
                
                if request.permission_ids:
                    permission_ids = await conn.fetch('''
                        SELECT id FROM permissions WHERE id = ANY($1)
                    ''', request.permission_ids)
                    permission_ids = [p['id'] for p in permission_ids]
                    
                    if len(permission_ids) != len(request.permission_ids):
                        logger.error(f"One or more permission IDs not found: {request.permission_ids}")
                        raise HTTPException(status_code=400, detail="One or more permission IDs not found")
                    
                    for perm_id in permission_ids:
                        await conn.execute('''
                            INSERT INTO role_permissions (role_id, permission_id)
                            VALUES ($1, $2)
                        ''', role_id, perm_id)
                
                await conn.execute('''
                    INSERT INTO audit_logs (user_id, action, details, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', current_user['id'], 'create_role', json.dumps({'role_name': request.name}), datetime.utcnow())
                
                logger.info(f"Role created: {request.name}")
                return {'message': 'Role created successfully', 'role_id': role_id}
            except asyncpg.UniqueViolationError:
                logger.error(f"Role name already exists: {request.name}")
                raise HTTPException(status_code=400, detail='Role name already exists')
            except Exception as e:
                logger.error(f"Error creating role {request.name}: {str(e)}")
                raise HTTPException(status_code=500, detail='Internal server error')

@app.put("/api/users/{user_id}/roles-permissions")
async def update_user_roles(user_id: int, request: UpdateUserRolesRequest, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Updating roles for user ID {user_id} by {current_user['username']}")
    await check_permission(current_user['id'], 'manage_users', db)
    async with db.acquire() as conn:
        async with conn.transaction():
            user = await conn.fetchrow('SELECT id FROM users WHERE id = $1', user_id)
            if not user:
                logger.error(f"User ID {user_id} not found")
                raise HTTPException(status_code=404, detail="User not found")
            
            await conn.execute('DELETE FROM user_roles WHERE user_id = $1', user_id)
            
            for role_id in request.role_ids:
                role = await conn.fetchrow('SELECT id FROM roles WHERE id = $1', role_id)
                if not role:
                    logger.error(f"Role ID {role_id} not found")
                    raise HTTPException(status_code=400, detail=f"Role ID {role_id} not found")
                await conn.execute('''
                    INSERT INTO user_roles (user_id, role_id)
                    VALUES ($1, $2)
                ''', user_id, role_id)
            
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'update_user_roles', json.dumps({'user_id': user_id, 'role_ids': request.role_ids}), datetime.utcnow())
            
            logger.info(f"User roles updated for user ID {user_id}")
            return {'message': 'User roles updated successfully'}

@app.get("/api/stats")
async def get_stats(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching stats by {current_user['username']}")
    await check_permission(current_user['id'], 'manage_users', db)
    async with db.acquire() as conn:
        stats = await conn.fetchrow('''
            SELECT 
                (SELECT COUNT(*) FROM users) as total_users,
                (SELECT COUNT(*) FROM users WHERE last_login IS NOT NULL) as active_users,
                (SELECT COUNT(*) FROM audit_logs WHERE action = 'login') as total_logins
        ''')
        logger.info(f"Stats retrieved: {stats}")
        return {
            'total_users': stats['total_users'],
            'active_users': stats['active_users'],
            'total_logins': stats['total_logins']
        }

@app.get("/api/audit-logs")
async def get_audit_logs(user_id: Optional[int] = None, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching audit logs by {current_user['username']}, user_id: {user_id}")
    await check_permission(current_user['id'], 'export_audit_logs', db)
    async with db.acquire() as conn:
        query = '''
            SELECT al.id, al.user_id, u.username, al.action, al.details, al.timestamp
            FROM audit_logs al
            JOIN users u ON al.user_id = u.id
        '''
        params = []
        if user_id is not None:
            query += ' WHERE al.user_id = $1'
            params.append(user_id)
        logs = await conn.fetch(query, *params)
        logger.info(f"Retrieved {len(logs)} audit logs")
        return [{
            'id': log['id'],
            'user_id': log['user_id'],
            'username': log['username'],
            'action': log['action'],
            'details': json.loads(log['details']),
            'timestamp': log['timestamp'].isoformat()
        } for log in logs]

@app.get("/api/config/orthanc", response_model=OrthancConfig)
async def get_orthogonal_config():
    settings = get_orthanc_settings()
    logger.info(f"Orthanc config retrieved: {settings['external_url']}")
    return {"orthanc_url": settings["external_url"]}

@app.get("/api/config/hospital", response_model=HospitalConfig)
async def get_hospital_config():
    hospital_config = {
        "hospital_name": os.getenv("HOSPITAL_NAME", "Sunrise Medical Center"),
        "hospital_address": os.getenv("HOSPITAL_ADDRESS", "123 Health St, Wellness City, HC 12345"),
        "hospital_phone": os.getenv("HOSPITAL_PHONE", "(123) 456-7890"),
        "hospital_logo": os.getenv("HOSPITAL_LOGO", "./logo.png")
    }
    logger.info(f"Hospital config retrieved: {hospital_config}")
    return hospital_config

@app.get("/api/orthanc/recent-studies")
async def get_recent_studies(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching recent studies by {current_user['username']}")
    await check_permission(current_user['id'], 'view_recent_studies', db)
    settings = get_orthanc_settings()
    try:
        test_response = requests.get(
            f"{settings['url']}/system",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        test_response.raise_for_status()
        logger.info(f"Orthanc system endpoint responded: {test_response.json().get('Version', 'Unknown')}")

        response = requests.get(
            f"{settings['url']}/studies",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        study_ids = response.json()

        if not study_ids:
            logger.info("No studies found in Orthanc")
            return []

        studies = []
        for study_id in study_ids[:10]:
            study_response = requests.get(
                f"{settings['url']}/studies/{study_id}",
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            study_response.raise_for_status()
            study = study_response.json()
            studies.append(study)

        sorted_studies = sorted(
            studies,
            key=lambda x: x.get("MainDicomTags", {}).get("StudyDate", "0"),
            reverse=True
        )[:10]

        logger.info(f"Retrieved {len(sorted_studies)} recent studies")
        return [{
            "id": study["ID"],
            "patient_name": study.get("PatientMainDicomTags", {}).get("PatientName", "Unknown Patient"),
            "patient_id": study.get("PatientMainDicomTags", {}).get("PatientID", "N/A"),
            "study_date": study.get("MainDicomTags", {}).get("StudyDate", "Unknown"),
            "study_description": study.get("MainDicomTags", {}).get("StudyDescription", "Unnamed Study"),
            "study_instance_uid": study.get("MainDicomTags", {}).get("StudyInstanceUID", study["ID"]),
            "modalities": study.get("MainDicomTags", {}).get("ModalitiesInStudy", "N/A"),
            "labels": study.get("Labels", [])
        } for study in sorted_studies]
    except requests.RequestException as e:
        logger.error(f"Error fetching recent studies: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent studies from Orthanc")
from datetime import datetime

@app.get("/api/orthanc/studies")
async def get_all_studies(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching all studies by {current_user['username']}")
    await check_permission(current_user['id'], 'view_studies', db)
    settings = get_orthanc_settings()
    try:
        # Test Orthanc connection
        test_response = requests.get(
            f"{settings['url']}/system",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        test_response.raise_for_status()
        logger.info(f"Orthanc system endpoint responded: {test_response.json().get('Version', 'Unknown')}")

        # Fetch all study IDs
        response = requests.get(
            f"{settings['url']}/studies",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        study_ids = response.json()

        if not study_ids:
            logger.info("No studies found in Orthanc")
            return []

        # Fetch details for each study
        studies = []
        for study_id in study_ids:
            study_response = requests.get(
                f"{settings['url']}/studies/{study_id}",
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            study_response.raise_for_status()
            study = study_response.json()

            studies.append(study)

        # Sort studies by date (newest first)
        sorted_studies = sorted(
            studies,
            key=lambda x: x.get("MainDicomTags", {}).get("StudyDate", "0"),
            reverse=True
        )

        logger.info(f"Retrieved {len(sorted_studies)} studies")
        return [{
            "id": study["ID"],
            "patient_name": study.get("PatientMainDicomTags", {}).get("PatientName", "Unknown Patient"),
            "patient_id": study.get("PatientMainDicomTags", {}).get("PatientID", "N/A"),
            "study_date": study.get("MainDicomTags", {}).get("StudyDate", "Unknown"),
            "study_description": study.get("MainDicomTags", {}).get("StudyDescription", "Unnamed Study"),
            "study_instance_uid": study.get("MainDicomTags", {}).get("StudyInstanceUID", study["ID"]),
            "modalities": study.get("MainDicomTags", {}).get("ModalitiesInStudy", "N/A"),
            "labels": study.get("Labels", []),
            "viewed": await check_study_viewed(study["ID"], db)
        } for study in sorted_studies]
    except requests.RequestException as e:
        logger.error(f"Error fetching all studies: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch studies from Orthanc")
    
# Helper function to check if a study has been viewed
async def check_study_viewed(study_id: str, db: asyncpg.Pool):
    async with db.acquire() as conn:
        view_count = await conn.fetchval('''
            SELECT COUNT(*) FROM study_views WHERE study_id = $1
        ''', study_id)
        return view_count > 0
    
@app.get("/api/orthanc/studies/search")
async def search_studies(
    query: str = "",
    page: int = 1,
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Pool = Depends(get_db)
):
    logger.info(f"Searching studies by {current_user['username']} with query: {query}, page: {page}, limit: {limit}")
    await check_permission(current_user['id'], 'search_studies', db)
    settings = get_orthanc_settings()

    try:
        # Test Orthanc connection
        test_response = requests.get(
            f"{settings['url']}/system",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        test_response.raise_for_status()

        # Construct Orthanc query
        orthanc_query = {
            "Level": "Study",
            "Query": {}
        }

        query = query.strip()
        if query:
            # Determine if query is likely a PatientID (numeric) or PatientName (alphanumeric)
            if query.isdigit():
                orthanc_query["Query"]["PatientID"] = f"{query}*"  # Exact match with trailing wildcard for PID
            else:
                orthanc_query["Query"]["PatientName"] = f"*{query}*"  # Wildcard for name
            orthanc_query["Query"]["StudyDate"] = "19000101-21001231"  # Broad date range
        else:
            orthanc_query["Query"]["PatientName"] = "*"
            orthanc_query["Query"]["PatientID"] = "*"
            orthanc_query["Query"]["StudyDescription"] = "*"

        # Log the exact query being sent
        logger.info(f"Sending Orthanc query: {json.dumps(orthanc_query)}")

        # Execute search
        find_response = requests.post(
            f"{settings['url']}/tools/find",
            json=orthanc_query,
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=10
        )
        find_response.raise_for_status()
        study_ids = find_response.json()

        if not study_ids:
            logger.info("No studies found for search query")
            return {"studies": [], "total": 0, "page": page, "limit": limit}

        # Paginate study IDs
        total = len(study_ids)
        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, total)
        paginated_study_ids = study_ids[start_idx:end_idx]

        # Fetch details for paginated studies
        studies = []
        for study_id in paginated_study_ids:
            study_response = requests.get(
                f"{settings['url']}/studies/{study_id}",
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            study_response.raise_for_status()
            study = study_response.json()
            studies.append({
                "id": study["ID"],
                "patient_name": study.get("PatientMainDicomTags", {}).get("PatientName", "Unknown Patient"),
                "patient_id": study.get("PatientMainDicomTags", {}).get("PatientID", "N/A"),
                "study_date": study.get("MainDicomTags", {}).get("StudyDate", "Unknown"),
                "study_description": study.get("MainDicomTags", {}).get("StudyDescription", "Unnamed Study")
            })

        logger.info(f"Retrieved {len(studies)} studies for page {page}")
        return {
            "studies": studies,
            "total": total,
            "page": page,
            "limit": limit
        }
    except requests.RequestException as e:
        logger.error(f"Error searching studies: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to search studies in Orthanc")
    
@app.get("/api/orthanc/studies/{study_id}")
async def get_study_details(study_id: str, current_user: dict = Depends(get_current_user)):
    logger.info(f"Fetching study {study_id} by {current_user['username']}")
    settings = get_orthanc_settings()
    try:
        test_response = requests.get(
            f"{settings['url']}/system",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        test_response.raise_for_status()
        logger.info(f"Orthanc system endpoint responded: {test_response.json().get('Version', 'Unknown')}")

        study_response = requests.get(
            f"{settings['url']}/studies/{study_id}",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        study_response.raise_for_status()
        study = study_response.json()

        logger.info(f"Study {study_id} details retrieved")
        return {
            "id": study["ID"],
            "patient_name": study.get("PatientMainDicomTags", {}).get("PatientName", "Unknown Patient"),
            "patient_id": study.get("PatientMainDicomTags", {}).get("PatientID", "N/A"),
            "study_date": study.get("MainDicomTags", {}).get("StudyDate", "Unknown"),
            "study_description": study.get("MainDicomTags", {}).get("StudyDescription", "Unnamed Study"),
            "study_instance_uid": study.get("MainDicomTags", {}).get("StudyInstanceUID", study["ID"]),
            "modalities": study.get("MainDicomTags", {}).get("ModalitiesInStudy", "N/A"),
            "labels": study.get("Labels", [])
        }
    except requests.RequestException as e:
        logger.error(f"Error fetching study {study_id}: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        if e.response and e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Study {study_id} not found in Orthanc")
        raise HTTPException(status_code=500, detail="Failed to fetch study details from Orthanc")

@app.post("/api/studies/view")
async def log_study_view(study_view: StudyView = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Logging study view for {study_view.study_id} in {study_view.viewer_type} by {current_user['username']}")
    await check_permission(current_user['id'], 'search_studies', db)
    async with db.acquire() as conn:
        try:
            await conn.execute('''
                INSERT INTO study_views (study_id, user_id, viewer_type, view_time)
                VALUES ($1, $2, $3, $4)
            ''', study_view.study_id, current_user['id'], study_view.viewer_type, datetime.utcnow())
            
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'view_study', json.dumps({
                'study_id': study_view.study_id,
                'viewer_type': study_view.viewer_type
            }), datetime.utcnow())
            
            logger.info(f"Study view logged for {study_view.study_id}")
            return {"message": "Study view logged"}
        except Exception as e:
            logger.error(f"Error logging study view for {study_view.study_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to log study view")

@app.get("/api/view/stone")
async def view_stone(study_instance_uid: str, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Viewing study {study_instance_uid} in Stone by {current_user['username']}")
    await check_permission(current_user['id'], 'search_studies', db)
    async with db.acquire() as conn:
        try:
            await conn.execute('''
                INSERT INTO study_views (study_id, user_id, viewer_type, view_time)
                VALUES ($1, $2, $3, $4)
            ''', study_instance_uid, current_user['id'], 'stone', datetime.utcnow())
            
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'view_study', json.dumps({
                'study_id': study_instance_uid,
                'viewer_type': 'stone'
            }), datetime.utcnow())
            
            settings = get_orthanc_settings()
            logger.info(f"Redirecting to Stone Viewer for study {study_instance_uid}")
            return RedirectResponse(f"{settings['external_url']}/stone-webviewer/index.html?study={study_instance_uid}")
        except Exception as e:
            logger.error(f"Error redirecting to Stone Viewer for {study_instance_uid}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to redirect to Stone Viewer")

@app.get("/api/view/ohif")
async def view_ohif(study_instance_uids: str, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Viewing study {study_instance_uids} in OHIF by {current_user['username']}")
    await check_permission(current_user['id'], 'search_studies', db)
    async with db.acquire() as conn:
        try:
            await conn.execute('''
                INSERT INTO study_views (study_id, user_id, viewer_type, view_time)
                VALUES ($1, $2, $3, $4)
            ''', study_instance_uids, current_user['id'], 'ohif', datetime.utcnow())
            
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'view_study', json.dumps({
                'study_id': study_instance_uids,
                'viewer_type': 'ohif'
            }), datetime.utcnow())
            
            logger.info(f"Redirecting to OHIF Viewer for study {study_instance_uids}")
            return RedirectResponse(f"/ohif/viewer?StudyInstanceUIDs={study_instance_uids}")
        except Exception as e:
            logger.error(f"Error redirecting to OHIF Viewer for {study_instance_uids}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to redirect to OHIF Viewer")

@app.post("/api/reports/start")
async def start_report(report_session: ReportSession = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Starting report for {report_session.study_id} by {current_user['username']}")
    await check_permission(current_user['id'], 'create_report', db)
    async with db.acquire() as conn:
        async with conn.transaction():
            try:
                # Check if a draft report already exists
                existing_report = await conn.fetchrow('''
                    SELECT id, status FROM reports
                    WHERE study_id = $1 AND created_by = $2 AND status = 'draft'
                ''', report_session.study_id, current_user['username'])

                if not existing_report:
                    # Fetch study details from Orthanc to populate the draft report
                    study_details = await get_study_details(report_session.study_id, current_user)
                    
                    # Parse study_date from YYYYMMDD format to a datetime.date object
                    study_date_str = study_details.get('study_date', datetime.utcnow().strftime("%Y%m%d"))
                    try:
                        if study_date_str != "Unknown":
                            study_date = datetime.strptime(study_date_str, "%Y%m%d").date()
                        else:
                            study_date = datetime.utcnow().date()
                    except ValueError:
                        logger.warning(f"Invalid study_date format for {study_date_str}, using current date as fallback")
                        study_date = datetime.utcnow().date()

                    # Create a draft report with minimal required fields
                    await conn.execute('''
                        INSERT INTO reports (
                            study_id, patient_name, patient_id, study_type, study_date,
                            study_description, findings, impression, radiologist_name,
                            report_date, created_by, created_at, status, start_time
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ''', report_session.study_id,
                        study_details.get('patient_name', 'Unknown Patient'),
                        study_details.get('patient_id', 'N/A'),
                        study_details.get('modalities', 'N/A'),
                        study_date,  # Now a datetime.date object
                        study_details.get('study_description', 'Unnamed Study'),
                        'Pending',  # Placeholder for findings
                        'Pending',  # Placeholder for impression
                        current_user['username'],  # Radiologist name
                        datetime.utcnow().strftime("%Y%m%d"),
                        current_user['username'],
                        datetime.utcnow(),
                        'draft',
                        datetime.utcnow())

                    logger.info(f"Draft report created for study {report_session.study_id}")

                # Create or update the report session
                await conn.execute('''
                    INSERT INTO report_sessions (study_id, user_id, start_time)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (study_id, user_id) DO UPDATE
                    SET start_time = EXCLUDED.start_time, end_time = NULL, completed = FALSE
                ''', report_session.study_id, current_user['id'], datetime.utcnow())

                # Log the action
                await conn.execute('''
                    INSERT INTO audit_logs (user_id, action, details, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', current_user['id'], 'start_report', json.dumps({
                    'study_id': report_session.study_id
                }), datetime.utcnow())

                logger.info(f"Report session started for {report_session.study_id}")
                return {"message": "Report session started"}
            except Exception as e:
                logger.error(f"Error starting report session for {report_session.study_id}: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to start report session")

@app.get("/api/reports", response_model=List[Report])
async def get_reports(status: Optional[str] = None, createdBy: Optional[str] = None, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching reports by {current_user['username']}, status: {status}, createdBy: {createdBy}")
    await check_permission(current_user['id'], 'view_reports', db)
    async with db.acquire() as conn:
        try:
            # Build the base query with corrected JOIN
            query = '''
                SELECT r.id, r.study_id, r.patient_name, r.patient_id, r.study_type, r.study_date,
                       r.study_description, r.findings, r.impression, r.recommendations,
                       r.radiologist_name, r.report_date, r.created_by, r.created_at,
                       r.status, r.notes, rs.start_time, rs.end_time
                FROM reports r
                LEFT JOIN users u ON r.created_by = u.username
                LEFT JOIN report_sessions rs ON r.study_id = rs.study_id AND rs.user_id = u.id
                WHERE 1=1
            '''
            params = []
            param_count = 1

            # Apply filters
            if status:
                query += f' AND r.status = ${param_count}'
                params.append(status)
                param_count += 1
            if createdBy:
                query += f' AND r.created_by = ${param_count}'
                params.append(createdBy)
                param_count += 1

            # Execute the query
            reports = await conn.fetch(query, *params)
            
            # Enrich reports with additional data
            enriched_reports = []
            for report in reports:
                # Fetch study details from Orthanc for modality and labels
                try:
                    study_response = await get_study_details(report['study_id'], current_user)
                    modality = study_response.get('modalities', 'N/A')
                    priority = study_response.get('labels', [])  # Assuming labels are used for priority
                    priority = ', '.join(priority) if priority else 'N/A'
                except HTTPException as e:
                    logger.warning(f"Failed to fetch study details for {report['study_id']}: {str(e)}")
                    modality = 'N/A'
                    priority = 'N/A'

                # Fetch viewer used from study_views
                viewer_used = await conn.fetchval('''
                    SELECT viewer_type
                    FROM study_views
                    WHERE study_id = $1
                    ORDER BY view_time DESC
                    LIMIT 1
                ''', report['study_id'])
                
                # Calculate duration if start_time and end_time exist
                duration = None
                if report['start_time'] and report['end_time']:
                    duration = (report['end_time'] - report['start_time']).total_seconds()
                
                enriched_reports.append({
                    'id': report['id'],
                    'studyId': report['study_id'],
                    'patientName': report['patient_name'],
                    'patientID': report['patient_id'],
                    'studyType': report['study_type'],
                    'studyDate': report['study_date'].isoformat() if report['study_date'] else None,
                    'studyDescription': report['study_description'],
                    'findings': report['findings'],
                    'impression': report['impression'],
                    'recommendations': report['recommendations'],
                    'radiologistName': report['radiologist_name'],
                    'reportDate': report['report_date'],
                    'createdBy': report['created_by'],
                    'createdAt': report['created_at'].isoformat() if report['created_at'] else None,
                    'status': report['status'],
                    'notes': report['notes'],
                    'startTime': report['start_time'].isoformat() if report['start_time'] else None,
                    'endTime': report['end_time'].isoformat() if report['end_time'] else None,
                    'duration': duration,
                    'modality': modality,
                    'priority': priority,
                    'viewerUsed': viewer_used if viewer_used else 'N/A'
                })
            
            logger.info(f"Retrieved {len(enriched_reports)} reports")
            return enriched_reports
        except Exception as e:
            logger.error(f"Error fetching reports: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch reports")

@app.post("/api/reports")
async def save_report(report: Report, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Saving report for study {report.studyId} by {current_user['username']}")
    await check_permission(current_user['id'], 'create_report', db)
    
    # Resolve the studyId to the internal Orthanc study ID
    internal_study_id = await resolve_study_id(report.studyId, current_user)
    logger.info(f"Using internal study ID {internal_study_id} for saving report")

    async with db.acquire() as conn:
        try:
            # Check if reports table exists
            table_exists = await conn.fetchval('''
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'reports'
                )
            ''')
            if not table_exists:
                logger.error("Reports table does not exist")
                raise HTTPException(status_code=500, detail="Reports table not found in database")

            # Insert or update the report using the internal study ID
            report_id = await conn.fetchval('''
                INSERT INTO reports (
                    study_id, patient_name, patient_id, study_type, study_date,
                    study_description, findings, impression, recommendations,
                    radiologist_name, report_date, created_by, created_at,
                    status, notes, ai_findings
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (study_id, created_by) DO UPDATE
                SET patient_name = EXCLUDED.patient_name,
                    patient_id = EXCLUDED.patient_id,
                    study_type = EXCLUDED.study_type,
                    study_date = EXCLUDED.study_date,
                    study_description = EXCLUDED.study_description,
                    findings = EXCLUDED.findings,
                    impression = EXCLUDED.impression,
                    recommendations = EXCLUDED.recommendations,
                    radiologist_name = EXCLUDED.radiologist_name,
                    report_date = EXCLUDED.report_date,
                    status = EXCLUDED.status,
                    notes = EXCLUDED.notes,
                    ai_findings = EXCLUDED.ai_findings,
                    created_at = EXCLUDED.created_at
                RETURNING id
            ''', internal_study_id, report.patientName, report.patientID, report.studyType,
                report.studyDate, report.studyDescription, report.findings,
                report.impression, report.recommendations, report.radiologistName,
                report.reportDate, current_user['username'], datetime.utcnow(),
                report.status, report.notes, json.dumps(report.ai_findings) if report.ai_findings else None)
            
            # Update report_sessions if exists
            if report.status == 'completed':
                await conn.execute('''
                    UPDATE report_sessions
                    SET end_time = $1, completed = TRUE
                    WHERE study_id = $2 AND user_id = $3 AND end_time IS NULL
                ''', datetime.utcnow(), internal_study_id, current_user['id'])
            
            # Log the action
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'create_report', json.dumps({
                'study_id': internal_study_id,
                'status': report.status
            }), datetime.utcnow())
            
            logger.info(f"Report saved successfully for study {internal_study_id}")
            return {"message": "Report saved successfully"}
        except asyncpg.InterfaceError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database connection error")
        except Exception as e:
            logger.error(f"Error saving report for study {internal_study_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")

@app.post("/api/reports/complete")
async def complete_report(complete: ReportComplete = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Completing report for study {complete.study_id} by {current_user['username']}")
    await check_permission(current_user['id'], 'create_report', db)
    
    # Resolve the study_id to the internal Orthanc study ID
    internal_study_id = await resolve_study_id(complete.study_id, current_user)
    logger.info(f"Using internal study ID {internal_study_id} for completing report")

    async with db.acquire() as conn:
        try:
            # Check if a report exists for the study and user
            report = await conn.fetchrow('''
                SELECT id, status FROM reports WHERE study_id = $1 AND created_by = $2
            ''', internal_study_id, current_user['username'])
            if not report:
                logger.error(f"No report found for study {internal_study_id} by {current_user['username']}")
                raise HTTPException(status_code=404, detail="No report found for the given study and user")

            # Update report status to completed if not already
            if report['status'] != 'completed':
                await conn.execute('''
                    UPDATE reports
                    SET status = 'completed'
                    WHERE id = $1
                ''', report['id'])

            # Update report session end time
            result = await conn.execute('''
                UPDATE report_sessions
                SET end_time = $1, completed = TRUE
                WHERE study_id = $2 AND user_id = $3 AND end_time IS NULL
            ''', datetime.utcnow(), internal_study_id, current_user['id'])
            if result == 'UPDATE 0':
                logger.warning(f"No active report session found for study {internal_study_id} by {current_user['username']}")

            # Log the completion action
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'complete_report', json.dumps({
                'study_id': internal_study_id
            }), datetime.utcnow())

            logger.info(f"Report completed successfully for study {internal_study_id}")
            return {"message": "Report completed successfully"}
        except Exception as e:
            logger.error(f"Error completing report for study {internal_study_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to complete report: {str(e)}")

@app.post("/api/studies/reassign")
async def reassign_study(request: ReassignStudy = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Reassigning study {request.study_id} to {request.new_doctor} by {current_user['username']}")
    await check_permission(current_user['id'], 'assign_studies', db)
    async with db.acquire() as conn:
        try:
            # Verify the new doctor exists
            new_doctor = await conn.fetchrow('''
                SELECT id, username
                FROM users
                WHERE username = $1
            ''', request.new_doctor)
            if not new_doctor:
                logger.error(f"Doctor not found: {request.new_doctor}")
                raise HTTPException(status_code=404, detail=f"Doctor {request.new_doctor} not found")

            # Update the report's created_by field
            updated = await conn.execute('''
                UPDATE reports
                SET created_by = $1
                WHERE study_id = $2 AND status = 'draft'
            ''', request.new_doctor, request.study_id)
            
            if updated == 'UPDATE 0':
                logger.error(f"No active report found for study {request.study_id} or report is not in draft status")
                raise HTTPException(status_code=404, detail="No active report found or report is not in draft status")

            # Update report_sessions if exists
            await conn.execute('''
                UPDATE report_sessions
                SET user_id = $1
                WHERE study_id = $2 AND completed = FALSE
            ''', new_doctor['id'], request.study_id)
            
            # Log the action
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'reassign_study', json.dumps({
                'study_id': request.study_id,
                'new_doctor': request.new_doctor
            }), datetime.utcnow())
            
            logger.info(f"Study {request.study_id} reassigned to {request.new_doctor}")
            return {"message": "Study reassigned successfully"}
        except Exception as e:
            logger.error(f"Error reassigning study {request.study_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to reassign study")

@app.get("/api/orthanc/modalities")
async def get_modalities(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching DICOM modalities by {current_user['username']}")
    await check_permission(current_user['id'], 'view_system', db)
    settings = get_orthanc_settings()
    try:
        response = requests.get(
            f"{settings['url']}/modalities",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        response.raise_for_status()
        modality_ids = response.json()
        
        modalities = []
        for modality_id in modality_ids:
            modality_response = requests.get(
                f"{settings['url']}/modalities/{modality_id}",
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            modality_response.raise_for_status()
            modality_info = modality_response.json()
            
            # Check modality status by attempting a query
            status = "Offline"
            try:
                query_response = requests.post(
                    f"{settings['url']}/modalities/{modality_id}/query",
                    json={"Level": "Study", "Query": {"StudyInstanceUID": "*"}},
                    auth=(settings["username"], settings["password"]),
                    timeout=3
                )
                if query_response.status_code == 200:
                    status = "Online"
            except requests.RequestException:
                pass
            
            modalities.append({
                "ID": modality_id,
                "AETitle": modality_info.get("AET", modality_id),
                "Host": modality_info.get("Host", "N/A"),
                "Port": modality_info.get("Port", "N/A"),
                "Status": status,
                "LastActivity": modality_info.get("LastActivity", "Unknown")
            })
        
        logger.info(f"Retrieved {len(modalities)} modalities")
        return modalities
    except requests.RequestException as e:
        logger.error(f"Error fetching modalities: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch modalities from Orthanc")

@app.post("/api/orthanc/search")
async def search_studies(raw_query: Dict[str, Any] = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Searching studies by {current_user['username']}: {raw_query}")
    await check_permission(current_user['id'], 'search_studies', db)
    settings = get_orthanc_settings()

    query_mapping = {
        "patientName": "patient_name",
        "patientId": "patient_id",
        "studyDateFrom": "study_date_from",
        "studyDateTo": "study_date_to",
        "studyDescription": "study_description",
        "accessionNumber": "accession_number",
        "referringPhysician": "referring_physician",
        "studyStatus": "study_status"
    }
    normalized_query = {}
    for key, value in raw_query.items():
        normalized_key = query_mapping.get(key, key)
        if normalized_key == "labels" and isinstance(value, str):
            normalized_query[normalized_key] = [label.strip() for label in value.split(",") if label.strip()] or None
        else:
            normalized_query[normalized_key] = value

    logger.info(f"Normalized search query: {normalized_query}")

    try:
        query = StudyQuery(**normalized_query)
        logger.info(f"Validated search query: {query.dict(exclude_unset=True)}")
    except ValidationError as e:
        logger.error(f"Validation error in search query: {e.errors()}")
        error_details = [
            f"Field '{err['loc'][0]}': {err['msg']}" for err in e.errors()
        ]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid search query: {'; '.join(error_details)}"
        )

    query_dict = query.dict(exclude_unset=True)
    if not query_dict:
        logger.info("Empty search query received")
        raise HTTPException(status_code=400, detail="At least one search criterion must be provided")

    try:
        test_response = requests.get(
            f"{settings['url']}/system",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        test_response.raise_for_status()
        logger.info(f"Orthanc system endpoint responded: {test_response.json().get('Version', 'Unknown')}")

        orthanc_query = {
            "Level": "Study",
            "Query": {}
        }
        if query.patient_name:
            orthanc_query["Query"]["PatientName"] = query.patient_name + "*"
        if query.patient_id:
            orthanc_query["Query"]["PatientID"] = query.patient_id + "*"
        if query.study_date_from or query.study_date_to:
            date_from = query.study_date_from.replace("-", "") if query.study_date_from else ""
            date_to = query.study_date_to.replace("-", "") if query.study_date_to else ""
            date_range = f"{date_from}-{date_to}".strip("-")
            orthanc_query["Query"]["StudyDate"] = date_range
        if query.modality:
            orthanc_query["Query"]["ModalitiesInStudy"] = query.modality
        if query.study_description:
            orthanc_query["Query"]["StudyDescription"] = query.study_description + "*"
        if query.accession_number:
            orthanc_query["Query"]["AccessionNumber"] = query.accession_number + "*"
        if query.referring_physician:
            orthanc_query["Query"]["ReferringPhysicianName"] = query.referring_physician + "*"
        if query.labels:
            orthanc_query["Labels"] = query.labels
        if query.study_status:
            orthanc_query["Labels"] = [query.study_status]

        logger.info(f"Constructed Orthanc query: {orthanc_query}")

        response = requests.post(
            f"{settings['url']}/tools/find",
            json=orthanc_query,
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        study_ids = response.json()
        logger.info(f"Orthanc /tools/find returned {len(study_ids)} study IDs")

        if not study_ids:
            logger.info("No studies found for search query")
            return []

        studies = []
        for study_id in study_ids:
            study_response = requests.get(
                f"{settings['url']}/studies/{study_id}",
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            study_response.raise_for_status()
            study = study_response.json()
            studies.append({
                "id": study["ID"],
                "patient_name": study.get("PatientMainDicomTags", {}).get("PatientName", "Unknown Patient"),
                "patient_id": study.get("PatientMainDicomTags", {}).get("PatientID", "N/A"),
                "study_date": study.get("MainDicomTags", {}).get("StudyDate", "Unknown"),
                "study_description": study.get("MainDicomTags", {}).get("StudyDescription", "Unnamed Study"),
                "study_instance_uid": study.get("MainDicomTags", {}).get("StudyInstanceUID", study["ID"]),
                "modalities": study.get("MainDicomTags", {}).get("ModalitiesInStudy", "N/A"),
                "labels": study.get("Labels", [])
            })
        logger.info(f"Retrieved details for {len(studies)} studies")
        return studies
    except requests.RequestException as e:
        logger.error(f"Error searching studies: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to search studies in Orthanc")

@app.post("/api/orthanc/tools/find")
async def orthanc_find(query: dict, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Orthanc find query by {current_user['username']}: {query}")
    await check_permission(current_user['id'], 'search_studies', db)
    settings = get_orthanc_settings()
    try:
        test_response = requests.get(
            f"{settings['url']}/system",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        test_response.raise_for_status()
        logger.info(f"Orthanc system endpoint responded: {test_response.json().get('Version', 'Unknown')}")

        response = requests.post(
            f"{settings['url']}/tools/find",
            json=query,
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Orthanc find returned {len(response.json())} results")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error in Orthanc find: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to query Orthanc")

@app.post("/api/orthanc/instances")
async def upload_instance(
    study_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Pool = Depends(get_db)
):
    logger.info(f"Received upload request for study {study_id} by {current_user['username']}")
    await check_permission(current_user['id'], 'upload_pdf', db)
    settings = get_orthanc_settings()

    if not file.filename.lower().endswith('.pdf'):
        logger.error(f"Invalid file type for upload: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Try to verify study_id as an internal Orthanc study ID
        logger.info(f"Verifying if {study_id} is an internal Orthanc study ID")
        study_response = requests.get(
            f"{settings['url']}/studies/{study_id}",
            auth=(settings["username"], settings["password"]),
            headers={"Accept": "application/json"},
            timeout=5
        )
        if study_response.status_code == 200:
            study = study_response.json()
            internal_study_id = study_id
            logger.info(f"Study {study_id} verified as internal Orthanc ID")
        else:
            # Assume study_id is a StudyInstanceUID and resolve to internal ID
            logger.info(f"Study {study_id} not found as internal ID, attempting to resolve as StudyInstanceUID")
            find_response = requests.post(
                f"{settings['url']}/tools/find",
                json={
                    "Level": "Study",
                    "Query": {
                        "StudyInstanceUID": study_id
                    }
                },
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            find_response.raise_for_status()
            study_ids = find_response.json()
            logger.info(f"Found {len(study_ids)} studies for StudyInstanceUID {study_id}")
            if not study_ids:
                logger.error(f"No study found for StudyInstanceUID {study_id}")
                raise HTTPException(status_code=404, detail=f"Study with StudyInstanceUID {study_id} not found in Orthanc")
            if len(study_ids) > 1:
                logger.warning(f"Multiple studies found for StudyInstanceUID {study_id}: {study_ids}")
            internal_study_id = study_ids[0]
            logger.info(f"Resolved StudyInstanceUID {study_id} to internal study ID {internal_study_id}")

            # Fetch study details with internal ID
            study_response = requests.get(
                f"{settings['url']}/studies/{internal_study_id}",
                auth=(settings["username"], settings["password"]),
                headers={"Accept": "application/json"},
                timeout=5
            )
            study_response.raise_for_status()
            study = study_response.json()
            logger.info(f"Study {internal_study_id} details retrieved")

        # Read file content
        file_content = await file.read()
        logger.info(f"Read PDF file, size: {len(file_content)} bytes")

        # Create a simple DICOM dataset to wrap the PDF
        dataset = pydicom.Dataset()
        dataset.file_meta = pydicom.dataset.FileMetaDataset()
        dataset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.104.1'  # Encapsulated PDF Storage
        dataset.file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.5.{uuid.uuid4()}"
        dataset.file_meta.ImplementationClassUID = '1.2.3.4'

        # Set required DICOM tags
        dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.104.1'
        dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study.get("MainDicomTags", {}).get("StudyInstanceUID", internal_study_id)
        dataset.SeriesInstanceUID = f"1.2.3.4.5.{uuid.uuid4()}"
        dataset.StudyID = study.get("MainDicomTags", {}).get("StudyID", "Unknown")
        dataset.SeriesNumber = "1"
        dataset.InstanceNumber = "1"
        dataset.Modality = "DOC"
        dataset.PatientName = study.get("PatientMainDicomTags", {}).get("PatientName", "Unknown Patient")
        dataset.PatientID = study.get("PatientMainDicomTags", {}).get("PatientID", "N/A")
        dataset.StudyDate = study.get("MainDicomTags", {}).get("StudyDate", datetime.utcnow().strftime("%Y%m%d"))
        dataset.StudyTime = study.get("MainDicomTags", {}).get("StudyTime", datetime.utcnow().strftime("%H%M%S"))
        dataset.AccessionNumber = study.get("MainDicomTags", {}).get("AccessionNumber", "")
        dataset.ReferringPhysicianName = study.get("MainDicomTags", {}).get("ReferringPhysicianName", "")

        # Encapsulate the PDF
        dataset.EncapsulatedDocument = file_content
        dataset.MIMETypeOfEncapsulatedDocument = "application/pdf"
        dataset.DocumentTitle = f"Report for Study {study_id}"

        # Save the dataset to a byte stream
        buffer = BytesIO()
        pydicom.filewriter.dcmwrite(buffer, dataset, write_like_original=False)
        dicom_data = buffer.getvalue()
        buffer.close()
        logger.info(f"Created DICOM dataset for PDF, size: {len(dicom_data)} bytes")

        # Upload to Orthanc
        upload_response = requests.post(
            f"{settings['url']}/instances",
            data=dicom_data,
            auth=(settings["username"], settings["password"]),
            headers={"Content-Type": "application/dicom"},
            timeout=10
        )
        upload_response.raise_for_status()
        instance_id = upload_response.json().get("ID")
        logger.info(f"Uploaded DICOM instance to Orthanc, instance ID: {instance_id}")

        # Log the action in audit logs
        async with db.acquire() as conn:
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'upload_pdf', json.dumps({
                'study_id': internal_study_id,
                'instance_id': instance_id,
                'filename': file.filename
            }), datetime.utcnow())

        logger.info(f"Successfully uploaded PDF for study {internal_study_id}")
        return {"message": "PDF uploaded successfully", "instance_id": instance_id}

    except requests.RequestException as e:
        logger.error(f"Error uploading PDF to Orthanc for study {study_id}: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Orthanc response: {e.response.text}")
        raise HTTPException(status_code=500, detail="Failed to upload PDF to Orthanc")
    except Exception as e:
        logger.error(f"Unexpected error while uploading PDF for study {study_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error while uploading PDF")
    
@app.get("/api/studies/unassigned")
async def get_unassigned_studies(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching unassigned studies by {current_user['username']}")
    await check_permission(current_user['id'], 'view_studies', db)
    async with db.acquire() as conn:
        studies = await conn.fetch('''
            SELECT r.study_id, r.patient_name, r.patient_id, r.study_description
            FROM reports r
            LEFT JOIN study_assignments sa ON r.study_id = sa.study_id
            WHERE sa.study_id IS NULL
        ''')
        logger.info(f"Retrieved {len(studies)} unassigned studies")
        return [{
            'study_id': s['study_id'],
            'patient_name': s['patient_name'],
            'patient_id': s['patient_id'],
            'study_description': s['study_description']
        } for s in studies]

@app.get("/api/studies/assigned")
async def get_assigned_studies(current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching assigned studies by {current_user['username']}")
    await check_permission(current_user['id'], 'view_assigned_studies', db)
    async with db.acquire() as conn:
        studies = await conn.fetch('''
            SELECT sa.study_id, u1.username AS assigned_to_username, u2.username AS assigned_by_username,
                   sa.assigned_on, sa.completion
            FROM study_assignments sa
            JOIN users u1 ON sa.assigned_to = u1.id
            JOIN users u2 ON sa.assigned_by = u2.id
        ''')
        logger.info(f"Retrieved {len(studies)} assigned studies")
        settings = get_orthanc_settings()
        assigned_studies = []
        for s in studies:
            # Fetch study details from Orthanc
            try:
                study_response = requests.get(
                    f"{settings['url']}/studies/{s['study_id']}",
                    auth=(settings["username"], settings["password"]),
                    headers={"Accept": "application/json"},
                    timeout=5
                )
                study_response.raise_for_status()
                study = study_response.json()
                assigned_studies.append({
                    'study_id': s['study_id'],
                    'patient_name': study.get("PatientMainDicomTags", {}).get("PatientName", "Unknown Patient"),
                    'patient_id': study.get("PatientMainDicomTags", {}).get("PatientID", "N/A"),
                    'study_description': study.get("MainDicomTags", {}).get("StudyDescription", "Unnamed Study"),
                    'assigned_to': s['assigned_to_username'],
                    'assigned_by': s['assigned_by_username'],
                    'assigned_on': s['assigned_on'].isoformat() if s['assigned_on'] else None,
                    'completion': s['completion']
                })
            except requests.RequestException as e:
                logger.error(f"Error fetching study {s['study_id']} from Orthanc: {str(e)}")
                assigned_studies.append({
                    'study_id': s['study_id'],
                    'patient_name': "Unknown Patient",
                    'patient_id': "N/A",
                    'study_description': "Unnamed Study",
                    'assigned_to': s['assigned_to_username'],
                    'assigned_by': s['assigned_by_username'],
                    'assigned_on': s['assigned_on'].isoformat() if s['assigned_on'] else None,
                    'completion': s['completion']
                })
        return assigned_studies

@app.post("/api/studies/assign")
async def assign_study(study_id: str = Body(...), assigned_to: str = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Assigning study {study_id} to {assigned_to} by {current_user['username']}")
    await check_permission(current_user['id'], 'assign_studies', db)
    internal_study_id = await resolve_study_id(study_id, current_user)
    async with db.acquire() as conn:
        async with conn.transaction():
            # Fetch or create user ID for assigned_to
            assigned_to_id = await conn.fetchval('SELECT id FROM users WHERE username = $1', assigned_to)
            if not assigned_to_id:
                logger.error(f"User {assigned_to} not found")
                raise HTTPException(status_code=404, detail=f"User {assigned_to} not found")

            # Insert or update assignment
            await conn.execute('''
                INSERT INTO study_assignments (study_id, assigned_to, assigned_by, assigned_on, completion)
                VALUES ($1, $2, $3, $4, '0%')
                ON CONFLICT ON CONSTRAINT unique_study_assignment DO UPDATE
                SET assigned_to = EXCLUDED.assigned_to,
                    assigned_by = EXCLUDED.assigned_by,
                    assigned_on = EXCLUDED.assigned_on,
                    completion = EXCLUDED.completion
            ''', internal_study_id, assigned_to_id, current_user['id'], datetime.utcnow())

            # Log the action
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'assign_study', json.dumps({
                'study_id': internal_study_id,
                'assigned_to': assigned_to
            }), datetime.utcnow())

            logger.info(f"Study {internal_study_id} assigned to {assigned_to}")
            return {"message": "Study assigned successfully"}
@app.put("/api/studies/reassign/{study_id}")
async def reassign_study(study_id: str, assigned_to: str = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Reassigning study {study_id} to {assigned_to} by {current_user['username']}")
    await check_permission(current_user['id'], 'assign_studies', db)
    internal_study_id = await resolve_study_id(study_id, current_user)
    async with db.acquire() as conn:
        async with conn.transaction():
            assigned_to_id = await conn.fetchval('SELECT id FROM users WHERE username = $1', assigned_to)
            if not assigned_to_id:
                logger.error(f"User {assigned_to} not found")
                raise HTTPException(status_code=404, detail=f"User {assigned_to} not found")
            await conn.execute('''
                UPDATE study_assignments
                SET assigned_to = $1, assigned_by = $2, assigned_on = $3, completion = '0%'
                WHERE study_id = $4
            ''', assigned_to_id, current_user['id'], datetime.utcnow(), internal_study_id)
            if conn.rowcount == 0:
                logger.error(f"No assignment found for study {internal_study_id}")
                raise HTTPException(status_code=404, detail=f"No assignment found for study {internal_study_id}")
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'reassign_study', json.dumps({
                'study_id': internal_study_id,
                'assigned_to': assigned_to
            }), datetime.utcnow())
            logger.info(f"Study {internal_study_id} reassigned to {assigned_to}")
            return {"message": "Study reassigned successfully"}

@app.delete("/api/studies/unassign/{study_id}")
async def unassign_study(study_id: str, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Unassigning study {study_id} by {current_user['username']}")
    await check_permission(current_user['id'], 'assign_studies', db)
    internal_study_id = await resolve_study_id(study_id, current_user)
    async with db.acquire() as conn:
        async with conn.transaction():
            result = await conn.execute('''
                DELETE FROM study_assignments WHERE study_id = $1
            ''', internal_study_id)
            if result == 'DELETE 0':
                logger.error(f"No assignment found for study {internal_study_id}")
                raise HTTPException(status_code=404, detail=f"No assignment found for study {internal_study_id}")
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'unassign_study', json.dumps({
                'study_id': internal_study_id
            }), datetime.utcnow())
            logger.info(f"Study {internal_study_id} unassigned")
            return {"message": "Study unassigned successfully"}

@app.post("/api/studies/update-status")
async def update_study_status(study_id: str = Body(...), status: str = Body(...), current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Updating status of study {study_id} to {status} by {current_user['username']}")
    await check_permission(current_user['id'], 'assign_studies', db)
    if status not in ['pending', 'in-progress', 'completed']:
        logger.error(f"Invalid status {status} provided")
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'pending', 'in-progress', or 'completed'.")
    
    # Map status to completion percentage
    completion_map = {
        'pending': '0%',
        'in-progress': '50%',
        'completed': '100%'
    }
    new_completion = completion_map[status]
    
    internal_study_id = await resolve_study_id(study_id, current_user)
    async with db.acquire() as conn:
        async with conn.transaction():
            # Update the study_assignments table
            result = await conn.execute('''
                UPDATE study_assignments
                SET completion = $1
                WHERE study_id = $2
            ''', new_completion, internal_study_id)
            
            if result == 'UPDATE 0':
                logger.error(f"No assignment found for study {internal_study_id}")
                raise HTTPException(status_code=404, detail=f"No assignment found for study {internal_study_id}")
            
            # Log the action
            await conn.execute('''
                INSERT INTO audit_logs (user_id, action, details, timestamp)
                VALUES ($1, $2, $3, $4)
            ''', current_user['id'], 'update_study_status', json.dumps({
                'study_id': internal_study_id,
                'status': status,
                'completion': new_completion
            }), datetime.utcnow())
            
            logger.info(f"Study {internal_study_id} status updated to {status}")
            return {"message": "Study status updated successfully"}

# Fetch emails for inbox, outbox, or sent
@app.get("/api/emails/{section}")
async def get_emails(section: str, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Fetching {section} emails for user ID {current_user['id']} (username: {current_user['username']})")
    await check_permission(current_user['id'], 'view_mails', db)

    if section not in ['inbox', 'outbox', 'sent']:
        logger.error(f"Invalid section: {section}")
        raise HTTPException(status_code=400, detail="Invalid section. Must be 'inbox', 'outbox', or 'sent'.")

    async with db.acquire() as conn:
        try:
            if section == 'inbox':
                query = '''
                    SELECT e.id, e.sender_id, u_sender.username AS sender, e.subject, e.body, e.timestamp,
                           e.is_external, e.is_read, e.status,
                           CASE WHEN e.is_external THEN 'external' ELSE 'internal' END AS recipient_type,
                           e.timestamp AS sent_at
                    FROM emails e
                    JOIN users u_sender ON e.sender_id = u_sender.id
                    WHERE (e.receiver_id = $1 OR e.receiver_email = (SELECT email FROM users WHERE id = $1))
                    AND e.status = 'sent'
                '''
            elif section == 'outbox':
                query = '''
                    SELECT e.id, e.sender_id, u_sender.username AS sender, e.subject, e.body, e.timestamp,
                           e.is_external, e.is_read, e.status,
                           CASE WHEN e.is_external THEN 'external' ELSE 'internal' END AS recipient_type,
                           e.timestamp AS sent_at
                    FROM emails e
                    JOIN users u_sender ON e.sender_id = u_sender.id
                    WHERE e.sender_id = $1 AND e.status = 'draft'
                '''
            else:  # sent
                query = '''
                    SELECT e.id, e.sender_id, u_sender.username AS sender, e.subject, e.body, e.timestamp,
                           e.is_external, e.is_read, e.status,
                           CASE WHEN e.is_external THEN 'external' ELSE 'internal' END AS recipient_type,
                           e.timestamp AS sent_at
                    FROM emails e
                    JOIN users u_sender ON e.sender_id = u_sender.id
                    WHERE e.sender_id = $1 AND e.status = 'sent'
                '''
            logger.debug(f"Executing query: {query} with params: {current_user['id']}")
            emails = await conn.fetch(query, current_user['id'])
            logger.debug(f"Fetched {len(emails)} emails: {[dict(email) for email in emails]}")
            return [dict(email) for email in emails]
        except Exception as e:
            logger.error(f"Error fetching {section} emails for user {current_user['username']}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch {section} emails: {str(e)}")
# Save a draft email
@app.post("/api/emails/draft")
async def save_draft(request: EmailRequest, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Saving draft by {current_user['username']}: {request.subject}")
    await check_permission(current_user['id'], 'send_emails', db)

    if request.recipient_type not in ['internal', 'external']:
        logger.error(f"Invalid recipient type: {request.recipient_type}")
        raise HTTPException(status_code=400, detail="Invalid recipient type. Must be 'internal' or 'external'.")

    async with db.acquire() as conn:
        async with conn.transaction():
            try:
                # Get sender's email
                sender_email = await conn.fetchval('SELECT email FROM users WHERE id = $1', current_user['id'])
                if not sender_email:
                    logger.error(f"Sender email not found for user {current_user['username']}")
                    raise HTTPException(status_code=400, detail="Sender email not configured")

                if request.recipient_type == 'internal':
                    # Find the recipient user
                    receiver = await conn.fetchrow('SELECT id, email FROM users WHERE username = $1', request.recipient)
                    if not receiver:
                        logger.error(f"Recipient not found: {request.recipient}")
                        raise HTTPException(status_code=404, detail="Recipient not found")
                    receiver_id = receiver['id']
                    receiver_email = None
                else:  # external
                    receiver_id = None
                    receiver_email = request.recipient

                # Insert the email as a draft
                email_id = await conn.fetchval('''
                    INSERT INTO emails (sender_id, receiver_id, receiver_email, subject, body, timestamp, status, is_external)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                ''', current_user['id'], receiver_id, receiver_email, request.subject, request.body,
                    datetime.utcnow(), 'draft', request.recipient_type == 'external')
                logger.info(f"Inserted email ID {email_id} with sender_id {current_user['id']}")

                # Log the action
                await conn.execute('''
                    INSERT INTO audit_logs (user_id, action, details, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', current_user['id'], 'save_draft', json.dumps({
                    'email_id': email_id,
                    'recipient': request.recipient,
                    'recipient_type': request.recipient_type,
                    'subject': request.subject
                }), datetime.utcnow())

                logger.info(f"Draft saved successfully by {current_user['username']}: {request.subject}")
                return {"message": "Draft saved successfully", "email_id": email_id}
            except Exception as e:
                logger.error(f"Error saving draft by {current_user['username']}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to save draft: {str(e)}")

# Send an email
@app.post("/api/emails/send")
async def send_email(request: EmailRequest, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Sending email by {current_user['username']}: {request.subject}")
    await check_permission(current_user['id'], 'send_emails', db)

    if request.recipient_type not in ['internal', 'external']:
        logger.error(f"Invalid recipient type: {request.recipient_type}")
        raise HTTPException(status_code=400, detail="Invalid recipient type. Must be 'internal' or 'external'.")

    async with db.acquire() as conn:
        async with conn.transaction():
            try:
                # Get sender's email
                sender_email = await conn.fetchval('SELECT email FROM users WHERE id = $1', current_user['id'])
                if not sender_email:
                    logger.error(f"Sender email not found for user {current_user['username']}")
                    raise HTTPException(status_code=400, detail="Sender email not configured")

                if request.recipient_type == 'internal':
                    receiver = await conn.fetchrow('SELECT id, email FROM users WHERE username = $1', request.recipient)
                    if not receiver:
                        logger.error(f"Recipient not found: {request.recipient}")
                        raise HTTPException(status_code=404, detail="Recipient not found")
                    receiver_id = receiver['id']
                    receiver_email = None
                else:  # external
                    receiver_id = None
                    receiver_email = request.recipient

                # Insert the email into the database
                email_id = await conn.fetchval('''
                    INSERT INTO emails (sender_id, receiver_id, receiver_email, subject, body, timestamp, status, is_external)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                ''', current_user['id'], receiver_id, receiver_email, request.subject, request.body,
                    datetime.utcnow(), 'sent', request.recipient_type == 'external')
                logger.info(f"Inserted email ID {email_id} with sender_id {current_user['id']}")

                # Verify the insertion
                inserted_email = await conn.fetchrow('SELECT * FROM emails WHERE id = $1', email_id)
                logger.debug(f"Inserted email data: {inserted_email}")

                

                # If external, send the email via MailerSend API
                if request.recipient_type == 'external':
                    api_token = os.getenv('MAILERSEND_API_TOKEN')
                    if not api_token:
                        logger.error("MailerSend API token missing")
                        raise HTTPException(status_code=500, detail="MailerSend API token missing")

                    mailer = emails.NewEmail(api_token)
                    mail_body = {
                        "from": {"email": sender_email},
                        "to": [{"email": receiver_email}],
                        "subject": request.subject,
                        "text": request.body
                    }

                    try:
                        response = mailer.send(mail_body)
                        # Debug: Log the raw response to check its type
                        logger.debug(f"MailerSend response: {response}")
                        if not hasattr(response, 'status_code'):
                            logger.error(f"Unexpected response from MailerSend: {type(response)} - {response}")
                            await conn.execute('UPDATE emails SET status = $2 WHERE id = $1', email_id, 'failed')
                            raise HTTPException(status_code=500, detail="Unexpected response from MailerSend")
                        if response.status_code != 202:
                            logger.error(f"MailerSend API error: {response.text}")
                            await conn.execute('UPDATE emails SET status = $2 WHERE id = $1', email_id, 'failed')
                            raise HTTPException(status_code=500, detail=f"MailerSend API error: {response.text}")
                        logger.info(f"External email sent to {receiver_email} via MailerSend")
                    except Exception as e:
                        logger.error(f"Failed to send external email to {receiver_email}: {str(e)}")
                        await conn.execute('UPDATE emails SET status = $2 WHERE id = $1', email_id, 'failed')
                        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

                # Log the action
                await conn.execute('''
                    INSERT INTO audit_logs (user_id, action, details, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', current_user['id'], 'send_email', json.dumps({
                    'email_id': email_id,
                    'recipient': request.recipient,
                    'recipient_type': request.recipient_type,
                    'subject': request.subject
                }), datetime.utcnow())

                logger.info(f"Email sent successfully by {current_user['username']}: {request.subject}")
                return {"message": "Email sent successfully", "email_id": email_id}
            except Exception as e:
                logger.error(f"Error sending email by {current_user['username']}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

# Mark emails as read
@app.post("/api/emails/mark-read")
async def mark_emails_as_read(request: EmailActionRequest, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Marking emails as read by {current_user['username']}: {request.email_ids}")
    await check_permission(current_user['id'], 'view_mails', db)

    async with db.acquire() as conn:
        async with conn.transaction():
            try:
                # Verify the emails belong to the user (inbox only)
                result = await conn.execute('''
                    UPDATE emails
                    SET is_read = TRUE
                    WHERE id = ANY($1::int[])
                    AND (receiver_id = $2 OR receiver_email = (SELECT email FROM users WHERE id = $2))
                    AND status = 'sent'
                ''', request.email_ids, current_user['id'])

                if result == 'UPDATE 0':
                    logger.error(f"No matching emails found to mark as read for user {current_user['username']}")
                    raise HTTPException(status_code=404, detail="No matching emails found to mark as read")

                # Log the action
                await conn.execute('''
                    INSERT INTO audit_logs (user_id, action, details, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', current_user['id'], 'mark_emails_read', json.dumps({
                    'email_ids': request.email_ids
                }), datetime.utcnow())

                logger.info(f"Emails marked as read by {current_user['username']}: {request.email_ids}")
                return {"message": "Emails marked as read successfully"}
            except Exception as e:
                logger.error(f"Error marking emails as read by {current_user['username']}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to mark emails as read: {str(e)}")

# Delete emails
@app.post("/api/emails/delete")
async def delete_emails(request: EmailActionRequest, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    logger.info(f"Deleting emails by {current_user['username']}: {request.email_ids}")
    await check_permission(current_user['id'], 'view_mails', db)

    async with db.acquire() as conn:
        async with conn.transaction():
            try:
                # Delete emails where the user is either the sender or receiver
                result = await conn.execute('''
                    DELETE FROM emails
                    WHERE id = ANY($1::int[])
                    AND (sender_id = $2 OR receiver_id = $2 OR receiver_email = (SELECT email FROM users WHERE id = $2))
                ''', request.email_ids, current_user['id'])

                if result == 'DELETE 0':
                    logger.error(f"No matching emails found to delete for user {current_user['username']}")
                    raise HTTPException(status_code=404, detail="No matching emails found to delete")

                # Log the action
                await conn.execute('''
                    INSERT INTO audit_logs (user_id, action, details, timestamp)
                    VALUES ($1, $2, $3, $4)
                ''', current_user['id'], 'delete_emails', json.dumps({
                    'email_ids': request.email_ids
                }), datetime.utcnow())

                logger.info(f"Emails deleted by {current_user['username']}: {request.email_ids}")
                return {"message": "Emails deleted successfully"}
            except Exception as e:
                logger.error(f"Error deleting emails by {current_user['username']}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to delete emails: {str(e)}")
            
@app.get("/api/ai/analyze/{study_id}", response_model=AIResult)
async def analyze_study(study_id: str, current_user: dict = Depends(get_current_user), db: asyncpg.Pool = Depends(get_db)):
    await check_permission(current_user['id'], 'use_ai', db)
    settings = get_orthanc_settings()
    internal_study_id = await resolve_study_id(study_id, current_user)

    async with db.acquire() as conn:
        instances_response = requests.get(
            f"{settings['url']}/studies/{internal_study_id}/instances",
            auth=(settings["username"], settings["password"]),
            timeout=5
        )
        instances_response.raise_for_status()
        instance_ids = instances_response.json()

        if not instance_ids:
            raise HTTPException(status_code=404, detail="No instances found for study")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 14)
        checkpoint = torch.load("resnet50_finetuned.pth", map_location=device, weights_only=True)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Model weights loaded successfully")
        model = model.to(device)
        model.eval()

        classes = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Pleural_Thickening"
        ]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        findings = []
        for instance in instance_ids:  # Removed [:1] to process all instances
            instance_id = instance['ID']
            instance_data = requests.get(
                f"{settings['url']}/instances/{instance_id}/file",
                auth=(settings["username"], settings["password"]),
                timeout=5
            )
            instance_data.raise_for_status()

            dicom = pydicom.dcmread(BytesIO(instance_data.content))
            image = dicom.pixel_array
            image = np.array(image, dtype=np.float32)
            if image.max() != image.min():
                image = (image - image.min()) / (image.max() - image.min())
            else:
                image = image / 255.0
            image = cv2.resize(image, (224, 224))
            image = np.stack((image,) * 3, axis=-1) if len(image.shape) == 2 else image

            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_tensor = transform(image_pil).unsqueeze(0).to(device)

            import time
            start_time = time.time()
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
                logger.info(f"Raw probabilities for {instance_id}: {probs}")
            inference_time = time.time() - start_time

            threshold = 0.3
            detected_conditions = [
                {"condition": label, "confidence": prob * 100}
                for label, prob in zip(classes, probs)
                if prob > threshold
            ]
            logger.info(f"Detected conditions for {instance_id}: {detected_conditions}")

            if not detected_conditions:
                detected_conditions.append({"condition": "Normal", "confidence": 100.0})

            max_prob_idx = np.argmax(probs)
            primary_condition = classes[max_prob_idx]
            primary_confidence = probs[max_prob_idx] * 100

            specific_findings = "No significant abnormalities detected." if detected_conditions[0]["condition"] == "Normal" else ", ".join(
                [f"Detected {cond['condition'].lower()} with {cond['confidence']:.2f}% confidence" for cond in detected_conditions]
            )

            if primary_confidence > 75:
                severity = "Severe"
            elif primary_confidence > 50:
                severity = "Moderate"
            else:
                severity = "Mild"

            if detected_conditions[0]["condition"] == "Normal":
                recommendations = "No immediate action required. Continue routine monitoring."
            else:
                recommendations = "Consult a specialist (e.g., pulmonologist or cardiologist) for further evaluation and consider follow-up imaging."

            supporting_metrics = {
                "primary_confidence": primary_confidence,
                "image_dimensions": "224x224 pixels",
                "inference_time": f"{inference_time:.2f} seconds",
                "model": "Fine-tuned ResNet50"
            }

            findings.append({
                "instance_id": instance_id,
                "detected_conditions": detected_conditions,
                "primary_condition": primary_condition,
                "primary_confidence": primary_confidence,
                "probabilities": {label: prob * 100 for label, prob in zip(classes, probs)},
                "specific_findings": specific_findings,
                "severity": severity,
                "recommendations": recommendations,
                "supporting_metrics": supporting_metrics
            })

        await conn.execute(
            "UPDATE reports SET ai_findings = $1 WHERE study_id = $2 AND created_by = $3",
            json.dumps(findings), internal_study_id, current_user['username']
        )

        await conn.execute(
            "INSERT INTO audit_logs (user_id, action, details, timestamp) VALUES ($1, $2, $3, $4)",
            current_user['id'], 'ai_analysis', json.dumps({"study_id": internal_study_id, "findings": findings}), datetime.utcnow()
        )

        return {
            "study_id": internal_study_id,
            "status": "completed",
            "result": findings[0] if findings else None,
            "specific_findings": findings[0]["specific_findings"] if findings else "No findings available",
            "recommendations": findings[0]["recommendations"] if findings else "No recommendations available",
            "supporting_metrics": findings[0]["supporting_metrics"] if findings else {},
            "severity": findings[0]["severity"] if findings else "Unknown"
        }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)