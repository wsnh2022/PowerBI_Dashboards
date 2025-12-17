# Power BI Desktop MySQL Setup Guide

Complete guide to installing Power BI Desktop and connecting to MySQL databases.

## Microsoft Store vs EXE Installation

**Use the standalone EXE installer, not Microsoft Store.**

### Why Store Version Causes Issues

Power BI from Microsoft Store is a UWP/MSIX package that:
- Runs in a sandbox with restricted permissions
- Doesn't appear in Control Panel or traditional uninstallers
- Has unreliable MySQL connector behavior
- Caches credentials incorrectly

### Why EXE Version Works Better

The standalone installer:
- Uses standard Win32 architecture
- Appears in Control Panel
- Loads database connectors reliably
- No sandbox restrictions

## Clean Uninstall Procedure

### 1. Remove Microsoft Store Version

**Method A: Microsoft Store**
1. Open Microsoft Store → Library
2. Find Power BI Desktop
3. Click Uninstall

**Method B: PowerShell** (if Store fails)
```powershell
Get-AppxPackage *PowerBI* | Remove-AppxPackage
```

### 2. Delete Local Data

Remove these folders:
```
C:\Users\<username>\Documents\Power BI Desktop
C:\Users\<username>\AppData\Local\Microsoft\Power BI
C:\Users\<username>\AppData\Local\Microsoft\Power BI Desktop
C:\Users\<username>\AppData\Roaming\Microsoft\Power BI
```

### 3. Clear Cached Credentials

1. Open Control Panel → Credential Manager
2. Windows Credentials
3. Delete entries containing:
   - Power BI
   - MySQL
   - localhost
   - ODBC

### 4. Reboot

Required to clear loaded DLLs and credential providers.

## Installation

### 1. Install Power BI Desktop

Download and install the standalone EXE from [Microsoft Power BI](https://powerbi.microsoft.com/desktop/).

### 2. Install MySQL Connector

**Required:** MySQL Connector/NET (64-bit, version 8.0.x)

Download from [MySQL Downloads](https://dev.mysql.com/downloads/connector/net/).

**Note:** Power BI does not use MySQL Workbench. It requires the standalone connector.

### 3. Reboot Again

Required after connector installation.

## MySQL Configuration

### Force Native Authentication

MySQL's default authentication plugin causes connection failures. Switch to native password:

```sql
ALTER USER 'root'@'localhost'
IDENTIFIED WITH mysql_native_password
BY 'your_password';

FLUSH PRIVILEGES;
```

Restart MySQL service after running this command.

## Connecting from Power BI

### Connection Steps

1. Power BI Desktop → Home → Get Data → MySQL database
2. Enter connection details:
   ```
   Server: localhost
   Database: your_database_name
   ```
3. Authentication settings:
   - Authentication kind: **Database**
   - **Use alternate credentials**
   - Username: MySQL username
   - Password: MySQL password
   - Apply level: `localhost:3306`
4. Click Connect

### Authentication Notes

- **Always use "Database" authentication**
- Do NOT use "Windows" authentication (MySQL doesn't support it)
- The UI may show "Windows authentication" but this is misleading

## Troubleshooting

### "Pre-login handshake" Error

**Cause:** Using SQL Server connector instead of MySQL connector

**Fix:** Use "MySQL database" connector, not "SQL Server"

### Credential Loop

**Cause:** Cached failed credentials

**Fix:** Clear Credential Manager (see step 3 under Clean Uninstall)

### "Windows authentication required" Prompt

**Cause:** Misleading UI text

**Fix:** Ignore the message, use Database authentication

## Verification

After setup, Power BI should:
- Load MySQL tables immediately
- Show no authentication prompts
- Connect reliably without errors

## Summary

**Root causes of connection issues:**
- Microsoft Store sandboxing
- Credential caching behavior
- MySQL authentication plugin mismatch

**Solution:**
- Clean EXE-based install
- Proper connector installation
- Native password authentication
- Complete credential reset
