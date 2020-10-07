import config
import json
from models import SentenceModel

class UserManagementRepositories:
    @staticmethod
    def create_users(userID,name,userName,password,email,phoneNo,roles):
        users = UserModel.create_users(userID,name,userName,password,email,phoneNo,roles)
        if users == None:
            return False
        return users
        
    @staticmethod
    def update_users(userID):
        for user in users:
            if UserModel.update_users_by_u_id(userID) == False:
                return False
        return True

    @staticmethod
    def search_users(userIDs,userNames,roleCodes):
        result = UserModel.get_user_by_keys(userIDs,userNames,roleCodes)
        # del result['_id']
        # del result['created_on']
        return result