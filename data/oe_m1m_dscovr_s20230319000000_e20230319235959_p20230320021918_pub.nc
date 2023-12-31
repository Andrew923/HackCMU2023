CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230319000000_e20230319235959_p20230320021918_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-20T02:19:18.138Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-19T00:00:00.000Z   time_coverage_end         2023-03-19T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxor�@  T          @����fff?c�
�q��6  C&+��fff@1G��-p���z�Ck�                                    Bxor��  �          @�G���ff?�{�33���C 
��ff@p��\(��ffC
                                    BxorČ  "          @�{�xQ�?W
=�O\)��C'� �xQ�@����\���
C�q                                    Bxor�2  T          @�p��}p�?
=�P  ��C+���}p�@\)����\)C�                                     Bxor��  
�          @���j=q?�  �J�H��\C$�j=q@"�\����͙�C=q                                    Bxor�~  
�          @���#�
?�\)�����X��C� �#�
@Tz��*�H�  B�8R                                    Bxor�$  T          @�p����R?����z��hffC}q���R@fff�'����B���                                    Bxos�  �          @�녾\)@z�������B�  �\)@����{����B��                                    Bxosp  
(          @�\)>�@O\)�p  �D��B�=q>�@����Q����RB��q                                    Bxos+  
�          @���=u@^{�Mp��+=qB��f=u@��Ϳ\(��'�
B�p�                                    Bxos9�  T          @�\)�L��@\(��N{�,G�B��3�L��@�(��aG��,��B���                                    BxosHb  
�          @��\�#�
@dz��Mp��'��B�#׾#�
@�\)�L����B��H                                    BxosW  T          @��׾k�@_\)�Mp��*Q�B��׾k�@�p��Y���%p�B��R                                    Bxose�  �          @����ff@333�O\)�+=qB��ff@��H���R�r=qB�G�                                    BxostT  
�          @�=q�p�@+��U�/z�Ck��p�@��������=qB�#�                                    Bxos��  �          @��R�(�?����h���O
=C�)�(�@`  ��֏\B�
=                                    Bxos��  
�          @���{������{��  CCG��{�>���*=q��C.�                                    Bxos�F  T          @�=q�J�H>�33�Vff�9�C-�H�J�H@��(���p�C��                                    Bxos��  �          @���G
=���R�G��+�HCI���G
=?�R�S33�8�\C(��                                    Bxos��  �          @���?   ��ff��33B�C�Ǯ?   ?�ff������B��\                                    Bxos�8  	�          @�녾��x���Q���z�C�� ���G����H�|\)C���                                    Bxos��  �          @���У��\)�ff��\Cw� �У�������c�\Ch�                                     Bxos�  �          @��H����|(��(���RC{+������\����n\)Cl�)                                    Bxos�*  �          @��׾�p���  �������HC��׾�p��Fff�`���Ap�C��
                                    Bxot�  "          @�\)�2�\�aG��n�R�T�C8n�2�\?�ff�Q��2ffC&f                                    Bxotv  �          @�{�;��c�
�tz��Mz�CDٚ�;�?��
�mp��E{CQ�                                    Bxot$  T          @�(��K���p��W��2��CI{�K�?@  �`  �<  C&�3                                    Bxot2�  
�          @�Q��(Q��33�[��:ffCW޸�(Q�>W
=�z�H�`(�C/c�                                    BxotAh  
�          @��H�,��?   �I���C�C)}q�,��@ff����p�C(�                                    BxotP  
�          @���=q?���~�R�ap�C�f�=q@B�\�2�\�  C u�                                    Bxot^�  �          @��Ϳٙ���  �����{C<5ÿٙ�@p��\)�\33B�L�                                    BxotmZ  T          @�{��p���G������ CVs3��p�?�z���p��z�Cff                                    Bxot|   �          @�z῝p���
=��G��C_��p�?��
�����)C�q                                    Bxot��  �          @�  ������H������CnO\���?O\)����33C\)                                    Bxot�L  "          @��
�e�@G�������HC���e�@L(��Q��\)C
O\                                    Bxot��  
�          @�G��U?�ff�I���%
=C��U@1G���p���{CO\                                    Bxot��  "          @��H�\(�?���=p���\C���\(�@=p���33���CG�                                    Bxot�>  T          @����s33?ٙ��#33����C�)�s33@2�\��G��o�C��                                    Bxot��  �          @�(��|��?�z��
=��\)CB��|��@.�R�G��  CY�                                    Bxot�  �          @�\)�fff@/\)��(���ffC���fff@W��k��'�C��                                    Bxot�0  "          @����Fff@c33�Ǯ��z�C!H�Fff@u?��@��
C �                                    Bxot��  "          @���J=q@z=q�����p  C ��J=q@`  ?��A�  C�                                    Bxou|  �          @���/\)@�=q>aG�@=qB����/\)@p��@&ffA�B�
=                                   Bxou"  �          @��H�Vff@�p��z���{B�\)�Vff@��?�A�=qC��                                   Bxou+�  "          @���S�
@��n{�=qC \)�S�
@�G�?�z�ArffCL�                                    Bxou:n  �          @�  �@  @h�������ffCu��@  @�Q�=L��?�\B�L�                                    BxouI  �          @����E�@xQ��=q���
C p��E�@�Q�>��H@�33B��q                                    BxouW�  
�          @����=p�@����˅���HB�u��=p�@���?G�A�
B�p�                                    Bxouf`  "          @����0��@���G����
B��0��@�33?k�A"=qB��R                                    Bxouu  �          @�z��2�\@����H�PQ�B�ff�2�\@���?�ffA`(�B��3                                    Bxou��  �          @�33�!G�@����
�H��  B��
�!G�@��\>�\)@C33Bힸ                                    Bxou�R  "          @�33��@a��N�R��B�  ��@�ff�fff�
=B��
                                    Bxou��            @������@#33��=q�W�B������@�{���\)B�aH                                    Bxou��  �          @��
�$z�@dz��7
=�
=B��=�$z�@��ÿ\)���
B�{                                    Bxou�D  
�          @�\)�8��@o\)�)����z�B�p��8��@�=q�����I��B��{                                    Bxou��  
Z          @����P  @�z��(���z�C &f�P  @�?5@�33B���                                    Bxouې  
�          @����Tz�@�(���
=�r=qC � �Tz�@���?p��A�\B��                                    Bxou�6  �          @�(��1G�@c33�.�R��{B����1G�@�ff������B�                                    Bxou��  �          @���Q�@E��g��+B�=q�Q�@��׿�(����\B�p�                                    Bxov�  �          @����,��@c�
�+����B�ff�,��@�{��
=��(�B�R                                    Bxov(  �          @�(��.�R@mp��333��Q�B��{�.�R@�(���ff���B�                                      Bxov$�  �          @�  �9��@I���Z�H���C���9��@�
=���\�V�RB��
                                    Bxov3t  "          @����W
=@Q��#33��z�C�R�W
=@�(��������C#�                                    BxovB  �          @�ff�N�R@_\)� ������C�q�N�R@���=�G�?�  C �R                                    BxovP�  "          @��
�c33@3�
<�>�
=C���c33@Q�?��RA�  C5�                                    Bxov_f  "          @��R�|��@.�R?�p�A�\)CW
�|��?�ff@G�B\)C!��                                    Bxovn  	�          @�p���Q�@K�>�
=@���C����Q�@   @ ��A���C�                                    Bxov|�  
�          @��H�e@]p��z�H�1C��e@]p�?uA/\)C
=                                    Bxov�X  "          @����:�H@0���-p��
=C�3�:�H@p�׿aG��'�B��                                    Bxov��  
�          @�Q��l(�@E��z���Q�C)�l(�@mp��#�
���C�
                                    Bxov��  S          @�z��s�
@,���ff����C�3�s�
@Y���Ǯ���
C
L�                                    Bxov�J  �          @�=q�w
=@.{������C޸�w
=@P  ������C��                                    Bxov��  
�          @�{�j�H@333�޸R���C���j�H@S33�#�
��ffC
{                                    BxovԖ  "          @���c�
@Mp���z���C	�R�c�
@n�R<�>�p�C��                                    Bxov�<  
�          @��R�Vff@^�R���R���RC���Vff@�  =�G�?�z�C�3                                    Bxov��  
�          @��
�c�
@:=q�  ��z�C���c�
@i�������z�C5�                                    Bxow �  �          @���]p�@5�.{��z�C�H�]p�@u�\(���HC�                                    Bxow.  
�          @��H�a�@J�H���Q�C
)�a�@z�H��p��}p�C                                      Bxow�  �          @�Q��[�@��?@  @�B����[�@aG�@@  A��CB�                                    Bxow,z  �          @���N�R@�Q�?Q�A=qB�8R�N�R@\(�@A�B�C5�                                    Bxow;   
Z          @�p��^�R@�ff>�p�@n{C ��^�R@g�@'
=A��
C��                                    BxowI�  
�          @��
�\(�@~�R�����8��C�f�\(�@}p�?���AB�HC�                                    BxowXl  
�          @����C33@�z῝p��W\)B�Ǯ�C33@�?�=qA<��B�=q                                    Bxowg  T          @�(��=p�@�33�E����B���=p�@��
?�=qA�=qB�aH                                    Bxowu�  
�          @�(��@��@�33�c�
�z�B�Q��@��@�{?�p�A�(�B�ff                                    Bxow�^  	�          @�ff�L��@��\�G��  B��)�L��@��
?ǮA�\)B��                                    Bxow�  T          @�
=�8Q�@}p����R��33B���8Q�@��?B�\A��B�k�                                    Bxow��  �          @�=q��
@\(��AG���B�����
@�  �Q��B�33                                    Bxow�P  �          @�����@��Ϳ�33�Tz�B�\)��@��
?��\Ak
=B�3                                    Bxow��  
�          @�ff��@0���o\)�IG�B�ff��@�G������33Bؙ�                                    Bxow͜  	�          @�{��@2�\�B�\� ��C33��@}p���
=�h  B���                                    Bxow�B  �          @����W�@X�ÿ����RC�
�W�@vff>��?�p�C(�                                    Bxow��  
(          @�G��9��@p  ?:�HA��B�G��9��@6ff@#33B ��CxR                                    Bxow��  	�          @�Q��   @~�R=�?���B�8R�   @W
=@��Aأ�B�G�                                    Bxox4  
�          @���=q@u����33B�L��=q@a�?ǮA���B���                                    Bxox�  
�          @���"�\@{����\�D��B��"�\@y��?���A[
=B�.                                    Bxox%�  �          @�33�G�@o\)��G�����C�=�G�@\)?�R@�{C �                                    Bxox4&  �          @�=q�   @q녿�����ffB����   @���?
=@�Q�B�B�                                    BxoxB�  
�          @�����@r�\������B�33��@��?z�@�G�B�                                    BxoxQr  	�          @�ff�AG�@Z=q��{���
C�=�AG�@xQ�>\)?˅B�                                    Bxox`  
Z          @��H�&ff@e��\�ř�B�  �&ff@��
=�\)?Y��B�\                                    Bxoxn�  
�          @�\)�#�
@g���\)��z�B�p��#�
@|(�>�ff@��RB��                                    Bxox}d  
�          @��H�%@��\�{�ř�B����%@�z�>#�
?޸RB�L�                                    Bxox�
  	�          @��
�9��@����  ��{B��H�9��@��H?Q�AffB��                                    Bxox��  
�          @�  �P��@�Q�h���\)C��P��@z=q?�  A]G�CǮ                                    Bxox�V  �          @��\�?\)@vff����~=qB����?\)@���?B�\A	��B�=q                                    Bxox��  "          @�\)�1�@b�\���\)C 33�1�@���#�
����B�8R                                    BxoxƢ  "          @�(��!�@��ÿ��H�[�B�{�!�@��?�\)AJ�\B�Ǯ                                    Bxox�H  �          @��
���@�G��xQ��/
=B�ff���@���?��HA���B�q                                    Bxox��  T          @�G��AG�@��R�u�((�B�L��AG�@��
?�ffAd��B��                                     Bxox�  �          @����=q@�=q����Q�B�=q�=q@��\>���@Mp�B���                                    Bxoy:  
�          @����@����33�̸RB����@��=�G�?�z�B�33                                    Bxoy�  "          @���g�@w
=����4z�C(��g�@w�?��A1�C�                                    Bxoy�  
�          @����8Q�@�  �(����B��=�8Q�@��=�?��B�\                                    Bxoy-,  �          @����]p�@R�\��\��33Cff�]p�@�  ���
�_\)C�
                                    Bxoy;�  "          @�
=�2�\@~{�"�\����B���2�\@��R�B�\�G�B�8R                                    BxoyJx  
�          @���Y��@8Q��?\)�Q�C�\�Y��@\)����C�C}q                                    BxoyY  �          @��
�k�@\)�>�R�\)C��k�@i����=q�ep�C@                                     Bxoyg�  �          @�\)�X��@�(���
=�F�RCp��X��@��?�ffA1��C5�                                    Bxoyvj  "          @�G��p  @p  �Ǯ����C���p  @���?�@���C�                                    Bxoy�  	�          @��\�y��@[���Q����C
���y��@|(�=#�
>�(�C�f                                    Bxoy��  T          @�p��e@]p��+���{C��e@��H�
=��{C�{                                    Bxoy�\  �          @�33���H@%��333���CǮ���H@hQ쿐���;�C
c�                                    Bxoy�  T          @�����
?�\�)����z�C����
@7
=��z��q�C�\                                    Bxoy��  
�          @�����  ?�\)�:�H���RC#
=��  @)���������C�                                    Bxoy�N  T          @��n�R@'��Vff�\)C��n�R@|(�������33Ch�                                    Bxoy��  T          @�(��q�@�
�fff���Ck��q�@e����G�C�\                                    Bxoy�  �          @���s33?����n�R�%�RC��s33@]p���
���
C	��                                    Bxoy�@  T          @�������?�Q��XQ����Cff����@W���(���  CB�                                    Bxoz�  
(          @�������?�z��`  �z�C �q����@=p����z�C��                                    Bxoz�  T          @���<(�?�p�����T(�CO\�<(�@J�H�Dz��Q�C�
                                    Bxoz&2  
�          @�p��(�?�ff�����w��Ch��(�@J�H�[��&�B�L�                                    Bxoz4�  
�          @�33���?����
=�y\)C�
���@Z�H�L(���B��                                    BxozC~  	�          @�z����?�����R�fC�����@O\)�dz��2B�u�                                    BxozR$  
Z          @�\)�-p�=��
��ff�jz�C2aH�-p�@G��u��={C
                                    Bxoz`�  "          @�(��$z�L�����R�u33C8p��$z�@
=q��ff�MQ�C                                    Bxozop  �          @�\)��(��W
=���H�
CN��(�?�z����
�z��C                                      Bxoz~  �          @���녿�����\)�z�RCU����?�z�������C\)                                    Bxoz��  �          @�\)���R�}p����
��CNc׿��R?���  �v{C�                                    Bxoz�b  �          @�\)�.{=��������h�
C1�)�.{@���q��;�C^�                                    Bxoz�  
�          @��R�O\)?
=q�~�R�I�HC*�)�O\)@��L���\)C0�                                    Bxoz��  	�          @���1녽�����p��gffC6��1�@�y���AG�C
                                    Bxoz�T  �          @��R��;�G�����o��C>(����?�p��~{�S��C��                                    Bxoz��  
�          @���aG�?�ff�j=q�.ffCJ=�aG�@I������  C
:�                                    Bxoz�  "          @�G��e�?�(��l(��1{C!0��e�@7��'���p�CE                                    Bxoz�F  �          @�Q��q�?�\)�QG��{C���q�@@����
����Cn                                    Bxo{�  
�          @�  �{�?�ff�8Q��z�Ck��{�@>�R�У���{C�H                                    Bxo{�  �          @��\����?�p��>{���C�)����@=p��޸R��
=C�
                                    Bxo{8  	�          @�=q�qG�@0  �p��ޏ\C���qG�@fff�O\)���CQ�                                    Bxo{-�  �          @����n{@N{��������C��n{@p�׾���{C�f                                    Bxo{<�  �          @�ff�dz�@{�S�
�p�CO\�dz�@qG���Q���(�Ch�                                    Bxo{K*  �          @���Tz�?�
=��  �8�RC�
�Tz�@hQ��#33�ۅCk�                                    Bxo{Y�  �          @��
�=p�?����~�R�CCn�=p�@a��&ff��p�C��                                    Bxo{hv  �          @�\)��@X���vff�3�
B���@��\��(�����Bݣ�                                    Bxo{w  
^          @�p����@333�z�H�:(�C����@�33�z���(�B��                                    Bxo{��  �          @���)��@AG��n�R�+ffCW
�)��@�ff��ff��
=B��                                    Bxo{�h  �          @�33�<��@>{�mp��&33C���<��@�z�����RB�                                    Bxo{�  �          @�p��A�@7
=�u��*p�C�3�A�@����(�����B��3                                    Bxo{��  
�          @�{�\)@2�\�����D(�C� �\)@�����H����B���                                    Bxo{�Z  
�          @�ff�(�@1���
=�N��C 0��(�@��H�$z��֏\B��                                    Bxo{�   
Z          @����\)@L(���{�B\)B����\)@�=q�Q���\)B�\)                                    Bxo{ݦ  �          @��H��
@u�QG���B�  ��
@�{��G��(z�B�\                                    Bxo{�L  T          @�\)�G�@���
=�T�\C�f�G�@���!G����HB�3                                    Bxo{��  
�          @�z��ff?���33�j�C
�ff@u��I����B�                                    Bxo|	�  "          @����
=@)����z��^z�B왚��
=@��%���  Bר�                                    Bxo|>  T          @��\��\@*=q��  �T33B�33��\@�(�����ۅB�                                    Bxo|&�  
�          @�33�Tz�@���\)�{Q�B��ͿTz�@����C�
�\)B��f                                    Bxo|5�  "          @�����@=q��(��l�B�k�����@��\�:=q��
Bօ                                    Bxo|D0  
�          @�Q��z�@3�
��(��U\)B�B���z�@��� ����ffB�
=                                    Bxo|R�  T          @�
=����@:=q���\�Q��B��q����@�(��=q���
B�B�                                    Bxo|a|  �          @��ÿ}p�@E��
=�XQ�B׏\�}p�@�33�p���\)B�{                                    Bxo|p"  "          @����  @*�H����g��B�(���  @����/\)����B�                                      Bxo|~�  
�          @�G���33?aG��K��n33C���33@�\�Q�� \)B�u�                                    Bxo|�n  �          @�  �N{@9���7
=��HC
�N{@z=q����K
=Cu�                                    Bxo|�  
�          @�z�����?����0  ����CG�����@:=q��ff���C�R                                    Bxo|��  "          @�z��XQ�>������K��C.W
�XQ�@��aG��#=qC                                      Bxo|�`  �          @�33��33����@  �	C::���33?�  �7���
C&�q                                    Bxo|�  T          @�z��z=q�xQ��c33�%z�CB  �z=q?Q��e�'�C(�                                    Bxo|֬  "          @�Q��p  ��  �U�� �\CF��p  >���a��,=qC,�
                                    Bxo|�R  
Z          @��R�U����R�N{�(CHp��U�>�(��Z�H�6G�C,��                                    Bxo|��  T          @�G���{@Q��l(��Z\)B��f��{@u�Q����\B�(�                                    Bxo}�  �          @��
�B�\��(��qG�=qCn5ÿB�\?333�z=qaHC	J=                                    Bxo}D  "          @�ff�z�@h���L(��#�B�{�z�@�
=�����P��B�                                    Bxo}�  
�          @�녾\@\)�Fff��B�녾\@�\)�\(��Q�B�                                    Bxo}.�  �          @�  �@  @�{�A����B�G��@  @��H�
=��{B�\                                    Bxo}=6  �          @�녿s33@w
=�w��1{Bϊ=�s33@�\)��{���HBȅ                                    Bxo}K�  "          @���>#�
@���hQ��'(�B���>#�
@�G���ff�]p�B���                                    Bxo}Z�  �          @��>�Q�@e�z�H�=�B��\>�Q�@�Q���
��ffB���                                    Bxo}i(  T          @����=q@w
=�n{�'
=B���=q@��Ϳ��R�v�RB�8R                                    Bxo}w�  
�          @���(�@x���W��33B�q�(�@�Q쿔z��@  B�.                                    Bxo}�t  S          @����(�@r�\�h���!��B��Ϳ�(�@��������o33Bޅ                                    Bxo}�  "          @����\)@aG��~�R�4  B����\)@�
=�����ffB�=q                                    Bxo}��  
�          @�(���ff@_\)��G��7z�B�{��ff@�
=���H���\B���                                    Bxo}�f  
�          @�녿˅@W
=����?�B晚�˅@�z�����{B�                                    Bxo}�  
�          @��ͼ��
@���  �RB�W
���
@���mp��$B���                                    Bxo}ϲ  
�          @�?@  ?�z������Bd��?@  @dz����\�Gp�B�aH                                    Bxo}�X  T          @�z�?��?c�
��=qB�?��@S33�����NQ�B��                                    Bxo}��  "          @�p�?�33?:�H����B(�?�33@L(������T�
B�=q                                    Bxo}��  
Z          @�p�?s33?=p����\)BG�?s33@L(������V��B��                                    Bxo~
J  "          @���?xQ�>u��p�#�A[�?xQ�@-p���ff�j  B���                                    Bxo~�  "          @�=q?��?�(�����u�Bk\)?��@y���q��,�\B�.                                    Bxo~'�  �          @�33?�G�?xQ������)B/�
?�G�@Vff��33�L��B�p�                                    Bxo~6<  �          @��?u?�������BB�?u@\(���Q��H=qB���                                    Bxo~D�  �          @�33?k�?Tz���p�G�B(z�?k�@N�R��{�S�B�=q                                    Bxo~S�  
�          @��H?��
<��
����Q�?=p�?��
@\)��G��n�RB{=q                                    Bxo~b.  
�          @�ff?p��?Tz���G�W
B&
=?p��@J=q��=q�R��B��                                    Bxo~p�  
�          @���?8Q�?�����HBf�H?8Q�@XQ����\�G
=B�                                      Bxo~z  "          @�ff?���?\��=q.BN��?���@j�H�s�
�2��B���                                    Bxo~�   "          @�G�?aG�@�\��G�p�B��=?aG�@���c33� �RB���                                    Bxo~��  �          @�Q�?�{?�\)���B�BXG�?�{@|(��e�"�HB�                                      Bxo~�l  "          @��H?��@�
��ff�z{B~��?��@���W
=��B�aH                                    Bxo~�  
�          @��?@  ?�  ���
B�B���?@  @x���qG��.\)B�33                                    Bxo~ȸ  "          @���?z�@G����G�B��?z�@��
�h���$�\B�
=                                    Bxo~�^  �          @���<�?޸R���(�B�<�@{��xQ��2p�B��                                    Bxo~�  T          @�����@?\)�����e  B����@����2�\��RB�33                                    Bxo~��  
�          @�G���@AG���(��cB�.��@�G��1G���ffB��q                                    BxoP  T          @�����@0����
=�n33B�𤾅�@��H�=p��{B��{                                    Bxo�  "          @��>k�@.{���\�r  B�aH>k�@���E����B�L�                                    Bxo �  "          @�녽��
@7
=��  �k��B�{���
@�{�<��� �RB��H                                    Bxo/B  
�          @�\)��Q�@z�����Bř���Q�@����U��\B��{                                    Bxo=�  
�          @�Q쾳33?�
=���R�B˽q��33@u�y���533B��                                     BxoL�  
�          @�Q쾣�
?n{����aHB�{���
@P  ����U��B�=q                                    Bxo[4  �          @�
=>��
?(���p�¥p�By��>��
@>�R��=q�c
=B��q                                    Bxoi�  �          @�\)=u>��
���R­8RB�p�=u@.�R����pQ�B�W
                                    Bxox�  T          @�  =u>\���R«��B�.=u@1����R�m��B�k�                                    Bxo�&  �          @���B�\��ff���R©��Cv�f�B�\@����HQ�B��f                                    Bxo��  �          @��?����H��Q�¤� C���?�?�z���8RB�G�                                    Bxo�r  
�          @��׾aG�?���P���t\)B��R�aG�@C33���{B�\                                    Bxo�  "          @����=q?������\ǮC ��=q@J=q�P���1�B�Ǯ                                    Bxo��  T          @�Q��B�\@\)�W��"p�C���B�\@n�R��z����C33                                    Bxo�d  T          @��
�p�?��R��{�`�C	�R�p�@p���Fff���B���                                    Bxo�
  
�          @�(��E�@/\)�U���RC
Y��E�@{���\��ffC {                                    Bxo��  
Z          @�z��A�@'
=�dz��&�C.�A�@z�H��\��\)B�Q�                                    Bxo�V  �          @���S33@!G��\(��p�C���S33@q녿�(���p�C
                                    Bxo�
�  %          @����@�\��
=�Rp�C޸��@z�H�1���z�B�\)                                    Bxo��  "          @�33��p�?�G�������C쿽p�@l(��a��&z�B߮                                    Bxo�(H  �          @���z�?c�
��k�C�\��z�@E���Q��QQ�B�k�                                    Bxo�6�  
�          @�p���@�
��
=�r��C ���@{��U�{B�\                                   Bxo�E�  
(          @����z�?��H���u�C uÿ�z�@a��z=q�9��B�k�                                    Bxo�T:  �          @�(���33?�p���{�t��C���33@u�Vff��
B➸                                    Bxo�b�  
�          @�
=�33?�\����p�CG��33@j=q�_\)��HB�{                                    Bxo�q�  
�          @��G�?�\)��33(�B��G�@k��xQ��7�B�                                    Bxo��,  "          @�p��J=q?aG������
C�f�J=q@E���
�Wz�BЏ\                                    Bxo���  �          @������?n{��{\C�ῌ��@E��Q��Q�B�                                      Bxo��x  �          @�=q�(�?����(��s  C�{�(�@P���c�
�(�
B��                                    Bxo��  T          @�(���?�p������y
=C�f��@Y���i���*�
B�{                                    Bxo���  �          @��\�Q�?�
=����m�HC���Q�@`���Z�H�(�B�                                    Bxo��j  "          @�
=���H?��
��\)\)CE���H@L���mp��6�\B�Q�                                    Bxo��            @��
=u?�G���z�B��=u@P  �w
=�G��B�
=                                    Bxo��  �          @�33�
=>�G����¢\)C��
=@(����H�i��B�Q�                                    Bxo��\  �          @��@  ��������i33C�Y�@  ?G����H�oQ�A�z�                                    Bxo�  
�          @���@>{�333�U����C�L�@>{������
=�T�C���                                    Bxo��  W          @��
?Ǯ�{��Q��d33C�3?Ǯ������z��RC��R                                    Bxo�!N  T          @��R>8Q�\���R(�C�G�>8Q�?\(����¡u�B��H                                    Bxo�/�  T          @�G��5�!G���{ .C]�׿5?�G���{�HB��H                                    Bxo�>�  �          @��Ϳ�(������  C5z῜(�@z���\)�x�B�Q�                                    Bxo�M@  �          @��׿�?5��G�{Cp���@@�����R�[
=B�z�                                    Bxo�[�  �          @��
�}p�@)�����\�iG�B�#׿}p�@�33�@���=qBͽq                                    Bxo�j�  T          @�z�z�H@6ff��p��_�B��f�z�H@��R�2�\���B̸R                                    Bxo�y2  T          @���@��@���c33�8{B6\)@��@n{�
�H��
=Bd�                                    Bxo���  �          @�\)�&ff@j�H�h���1z�B�33�&ff@��
�޸R��(�B�G�                                    Bxo��~  
�          @�zῂ�\@Z=q��Q��A�B�\)���\@�G�����=qB��                                    Bxo��$  T          @�=q��  @r�\�Z�H�"p�B؞���  @�(���  ��=qB���                                    Bxo���  Q          @�=q�   ?Ǯ��G�Q�B��
�   @U��]p��6�HB�33                                    Bxo��p  W          @��
�\)�O\)��p�� Ck^��\)?���������B�{                                    Bxo��  T          @��\�c�
?E���(��C!H�c�
@7
=���\�]�B֞�                                    Bxo�߼  �          @��}p�?�ff������CB��}p�@G���\)�Q  B�.                                    Bxo��b  �          @�\)�   >�z�����=C+�f�   @����
=�\z�C�{                                    Bxo��  "          @����   �Ǯ��z�ǮC?+��   ?�ff���\�s��C
                                      Bxo��  
�          @�����R?
=��  aHC%5���R@(Q���G��L��C\)                                    Bxo�T  �          @����?�����\�r�RC&f��@Q��s33�-�\B��                                    Bxo�(�  
�          @������?���
=�l(�C�����@P���l���(�RB��
                                    Bxo�7�  �          @���,��?�=q����j(�C:��,��@=p��w
=�/��C\)                                    Bxo�FF  
�          @�(��+�?���G��Vz�C&f�+�@S33�L���  C)                                    Bxo�T�  
�          @�ff�   ?�=q�����cQ�C�R�   @Tz��]p���B�                                      Bxo�c�  "          @�����?�������zG�C����@O\)�z�H�4p�B��H                                    Bxo�r8  
Z          @���	��?��������CL��	��@Fff��=q�<�B��                                    Bxo���  T          @�33�
=?����{�y�\C5��
=@?\)��=q�;Q�C O\                                    Bxo���  "          @����=q?�  ��z��wz�Cn�=q@;���G��;  Cs3                                    Bxo��*  
�          @���:=q?^�R��(��b��C#\)�:=q@,���w
=�0��C	&f                                    Bxo���  �          @�33�XQ�>u���RQ�C/�f�XQ�@33�|(��3��C�{                                    Bxo��v  �          @�(��xQ�?�R�{��3�HC*���xQ�@��U��  C�                                    Bxo��  "          @�G�����@�������\C+�����@Q�?!G�@��C��                                    Bxo���  �          @����{@s�
?�A�{C	����{@.{@O\)B��C�R                                    Bxo��h            @�Q�����@z=q?�G�ADz�C	� ����@C�
@0  A�G�Ck�                                    Bxo��  
�          @�p��
=q�L������¨��C9&f�
=q@
=q��ff�B��H                                    Bxo��  "          @�33?@  ��=q��z���C���?@  ?�=q��{�B]�                                    Bxo�Z  
�          @�G�?z�\����Q�C�n?z�?^�R��(�Ba�\                                    Bxo�"   
�          @�����
?fff���
�up�Cp���
@   �Y���9�C��                                    Bxo�0�  �          @�����Q�@Mp����\)C�{��Q�@xQ�W
=�   CJ=                                    Bxo�?L  T          @�=q����@@���(Q���C������@tzῗ
=�4(�C                                      Bxo�M�  "          @�(���
=@@��� ����p�Ck���
=@qG���=q�"{CW
                                    Bxo�\�  T          @�p���33@/\)�\(���CǮ��33@y����\��  C
#�                                    Bxo�k>  �          @�z���{@/\)�c�
���C����{@|(��	����  C�                                     Bxo�y�  �          @�33����@(���p  ��\C����@{�������C�q                                    Bxo���  "          @�\)�n{@#33�z�H�${C�
�n{@z=q�$z��˅C��                                    Bxo��0  �          @���c33@�����0�C�
�c33@y���5���{CT{                                    Bxo���  
(          @����E�@ff���R�D{C���E�@|(��I��� �C �                                    Bxo��|  
(          @�{�Z=q@�����;(�Cn�Z=q@n�R�E�����C��                                    Bxo��"  �          @��
�(Q�@�����
�Pz�C�R�(Q�@�33�P���33B�aH                                    Bxo���  "          @����/\)@p����\�L�RC

�/\)@��\�N{���B�                                    Bxo��n  T          @�(�� ��@����\)�V��C@ � ��@��H�XQ����B�#�                                    Bxo��  
�          @����=q@���p��cz�C
���=q@y���j�H�B�                                    Bxo���  
�          @�=q�HQ�@hQ��p�����C�R�HQ�@��\� ����ffB��)                                    Bxo�`  
�          @��H�C33@z=q�`���33B��f�C33@�\)��
=��  B���                                    Bxo�  T          @�=q�_\)@Vff�h�����C(��_\)@�Q��G���  B�W
                                    Bxo�)�  "          @���`��@N{�o\)���C	p��`��@�{�
�H����C T{                                    Bxo�8R  �          @�(��X��@tz��W
=�=qC�
�X��@��H��=q�o�B�{                                    Bxo�F�  
�          @\�qG�@<���o\)�{C���qG�@�����=qC                                    Bxo�U�  �          @�=q�mp�@P���_\)��C
���mp�@��
��
=���CY�                                    Bxo�dD  �          @�G��z�H@Q��J=q��  C\�z�H@�Q��\)�z�\C��                                    Bxo�r�  
(          @Å�vff@�������(�C�\�vff@��&ff��z�Cu�                                    Bxo���  "          @Å��\)@�p���\)�O�Cs3��\)@��
>�p�@]p�C{                                    Bxo��6  �          @�(����@����\)�(z�CW
���@�Q�?
=@���C�)                                    Bxo���  
Z          @������@�Q쿜(��9�C�3���@���?
=q@��
C
=                                    Bxo���  T          @�\)�^{@dz��,(���33C:��^{@��\����2{C ��                                    Bxo��(  
�          @�33�7�@J=q���
�6�CJ=�7�@���333��=qB��H                                    Bxo���  
Z          @�����Q�@��\��\)�,(�CB���Q�@w�?�=qAO�C	Ǯ                                    Bxo��t  
Z          @�33��ff@s33?�
=A�C����ff@7
=@AG�A�CQ�                                    Bxo��  
�          @����  @g������G�C���  @a�?uA�
Cn                                    Bxo���  
�          @��H��Q�@q녽L�Ϳ   C�\��Q�@`��?�z�AVffC��                                    Bxo�f  
�          @\���@~{>B�\?��C
����@e?ٙ�A��HC�                                     Bxo�  
�          @\��\)@�=q?�@�(�Cc���\)@r�\@��A���C
!H                                    Bxo�"�  
(          @Å��33@xQ�?.{@�{Cٚ��33@Tz�@�A���C(�                                    Bxo�1X  T          @�{���@�>�\)@$z�C	�����@p  ?�{A���C��                                    Bxo�?�  �          @�\)��\)@�G��.{��{C}q��\)@s33?���AM��C5�                                    Bxo�N�  �          @�����  @�33��\)��RC+���  @tz�?�  A]C33                                    Bxo�]J  �          @�=q��{@��R�u�	��C
���{@~�R?��AJ�HC��                                    Bxo�k�  
�          @��
���@�ff�(�����C
� ���@�(�?}p�A33C
�3                                    Bxo�z�  T          @�33���
@~{�k��p�C�\���
@���?&ff@�p�Cp�                                    Bxo��<  
�          @Ǯ���@|�Ϳc�
�\)C.���@\)?+�@�ffC�H                                    Bxo���  
\          @�=q��G�@j=q�u�\)C���G�@p  ?   @�=qCY�                                    Bxo���  
�          @�Q�����@h�ÿ#�
��z�C\����@g�?G�@�RCE                                    Bxo��.  �          @������\@j�H�z�H�
=C�q���\@qG�>�@�ffC                                      Bxo���  
�          @�{���
@k��h���  C�R���
@p  ?
=q@�Q�Ck�                                    Bxo��z  �          @�{��  @p  ��p��733C����  @|(�>�=q@!�C^�                                    Bxo��   �          @���33@S33����� (�C)��33@]p�>u@��Cٚ                                    Bxo���  "          @����
@P�׿�{�%��C����
@\(�>B�\?��C!H                                    Bxo��l  
�          @�����@N�R��{�&ffC�H����@Z=q>.{?�{Cs3                                    Bxo�  $          @������@x�ÿ�Q��^�RC	�H����@�z�=�?���C\                                    Bxo��  
Z          @����@��\�Ǯ�o33C����@��=�\)?.{C                                    Bxo�*^  "          @����@z�H���\�B=qC
���@��>�\)@'�C	(�                                    Bxo�9  T          @�Q����\@l(���\)�-G�C
���\@vff>���@HQ�C�                                    Bxo�G�  
(          @�Q���z�@g���33�0��C\��z�@r�\>�=q@'
=C�=                                    Bxo�VP  T          @�Q����@A녿����-G�C
���@N�R=�\)?0��Cn                                    Bxo�d�  T          @�Q����R@1녿���'33C�����R@?\)<��
>��C(�                                    Bxo�s�  T          @�  ����@C�
�����-�C������@P��=��
?=p�C                                    Bxo��B  �          @����{@"�\�!G����C����{@&ff>�33@Tz�C�                                     Bxo���  �          @\��Q�@�R�����ffC����Q�@!�>�33@Tz�CO\                                    Bxo���  
�          @�G����@7��B�\��  CB����@<��>�33@S33C��                                    Bxo��4  �          @������?�녿�33�/\)C!�����@��\�g
=C�f                                    Bxo���  "          @�����z�@�\������RC )��z�@#33�k���HC�f                                    Bxo�ˀ  �          @ȣ���  ?�  ��(���
C#���  @�׿u��C��                                    Bxo��&  �          @�Q����?��
�p���
=qC#=q���@   �k��z�C!O\                                    Bxo���  "          @ə����H?��þu��RC%�\���H?�ff>�{@G
=C%�R                                    Bxo��r  
(          @�  ��33?�ff�����C'�R��33?�(�>�ff@��HC(��                                    Bxo�  �          @Ǯ��?B�\��(��~�RC,�3��?^�R����C+�q                                    Bxo��  
(          @��H��?:�H��
=�,  C-B���?�{�W
=��33C)�
                                    Bxo�#d             @������?   ����<Q�C/^�����?n{���\���C+k�                                    Bxo�2
  V          @˅���
?녿�ff�b�RC.�����
?�����R�4��C)�)                                    Bxo�@�  �          @�=q��
=?^�R������ffC+�3��
=?�(�����K\)C&0�                                    Bxo�OV  T          @�=q���?xQ��   ���C*�3���?�\)�\�^�RC$��                                    Bxo�]�  
�          @�=q���>�  �z���ffC1�����?�������C)                                    Bxo�l�  �          @ə���>Ǯ� ����G�C0E��?�=q�޸R���\C)��                                    Bxo�{H  
�          @�����{>��
��Q����RC0���{?}p��ٙ��|z�C*�=                                    Bxo���  "          @�\)���
>Ǯ�G����C0(����
?����  ����C)}q                                    Bxo���  T          @�
=���>u�"�\���C1�{���?�����\����C(�q                                    Bxo��:  �          @�G����\?�z���
����C#}q���\@������hQ�C�                                    Bxo���  
�          @�  ���?����!G���  C%�f���@���\)��Q�CQ�                                    Bxo�Ć  "          @�ff���?fff�(���G�C*�{���?ٙ����H��G�C"�R                                    Bxo��,  �          @�������@Q���R����C0�����@;����\���CY�                                    Bxo���  T          @�G�����?�  �
=���\C)����?�
=��\)�z�RC#�                                    Bxo��x  
�          @�����ff?W
=������  C+G���ff?�(�����s33C$��                                    Bxo��  
�          @�p���>B�\�˅�o�C20���?:�H���U�C,�3                                    Bxo��  
�          @�(���p��k���(��_33C6@ ��p�>������H�]G�C1)                                    Bxo�j  "          @Å��
=>�Q쿁G��G�C0�=��
=?0�׿Q���Q�C-ff                                    Bxo�+  T          @��H���\@
=�����'
=CL����\@
=��z��1�C{                                    Bxo�9�  �          @��
��
=?�׿����C!�
��
=@Q쾮{�J�HC��                                    Bxo�H\  
�          @����ff@ �׿�\)�'�C �=��ff@녾�33�N�RC:�                                    Bxo�W  "          @�p���33@�ÿs33��\C�f��33@$z὏\)�(�CW
                                    Bxo�e�  
Z          @�����@�h�����Cc����@ �׽L�;�
=C��                                    Bxo�tN  T          @����{@
=��Q��Z{C�\��{@-p�����Q�C�                                    Bxo���  T          @����ff@J�H��(����\CT{��ff@dz����33C.                                    Bxo���  
�          @�z�����@U���=q�"�RC������@_\)>�?��RC0�                                    Bxo��@             @�����@W
=�B�\���
Cz����@Z=q>�G�@���C
=                                    Bxo���  
�          @������@,��=�\)?#�
C�����@ ��?}p�AQ�C�)                                    Bxo���  �          @�{���@'�>��H@��HC�=���@�\?�{AL(�C��                                    Bxo��2  T          @�z���z�@:=q����Q�C�H��z�@1�?aG�A�C�q                                    Bxo���  "          @���p�@E�����RC���p�@?\)?O\)@��C�                                    Bxo��~  T          @�{�vff@�=q�$z���33CY��vff@��R���
�G�C@                                     Bxo��$  "          @���s�
@x���$z����CW
�s�
@�G�����&�\C                                      Bxo��  T          @����\?�ff@\)B!G�C'#����\��p�@��B'�C8��                                    Bxo�p  "          @�\)��33?k�@��B1�HC(���33�(�@��B4��C<�                                    Bxo�$  "          @�Q���(�?���@�  B+33C"u���(��\)@�
=B6Q�C5��                                    Bxo�2�  �          @\���\?h��@��B:{C'k����\�!G�@�ffB<C<�{                                    Bxo�Ab  �          @�=q���?�ff@�z�B9(�C%}q��녾��H@��B>�\C:��                                    Bxo�P  �          @�=q��=q@
�H@[�Bz�C^���=q?aG�@}p�B'�RC(p�                                    Bxo�^�  
�          @\���@�@C�
A��
C�����?�\)@h��B�C&ff                                    Bxo�mT  V          @�\)��
=@
=@P��A�Cn��
=?���@w
=B�C&��                                    Bxo�{�  "          @ƸR��
=@ff@P��B
=C�R��
=?c�
@q�BC)T{                                    Bxo���  
�          @�  ��?�Q�@S33A��C����?8Q�@p��B33C+��                                    Bxo��F  
�          @ȣ����?�z�@HQ�A�(�C"!H���?�@`��BQ�C.8R                                    Bxo���  �          @����?��
@H��A�z�C!���?!G�@c33B��C-
=                                    Bxo���  �          @�=q���R@G�@(�A��C�����R?��@>{A�z�C(�3                                    Bxo��8  "          @˅��(�@33?��A�  C�=��(�?��@"�\A�ffC$��                                    Bxo���  �          @�����{@G�?�\A�
=CE��{?�ff@�HA��RC$�q                                    Bxo��  �          @�p����\@:=q?�  A4Q�C� ���\@�@	��A�=qC^�                                    Bxo��*  "          @ə�����@N{?�p�A4(�C������@(��@�RA��Cp�                                    Bxo���  
(          @�33���@Y��?.{@�z�Cc����@?\)?�G�A���C��                                    Bxo�v  	�          @�33���@p  >\@\(�C�����@Z�H?˅Ah��C8R                                    Bxo�  "          @������@xQ�>�ff@��C�f���@aG�?ٙ�A{�CY�                                    Bxo�+�  T          @˅��{@~{>��
@8��C0���{@i��?�{AjffC��                                    Bxo�:h  �          @Ǯ����@~{>�=q@p�Cp�����@j=q?�ffAf�\C�                                    Bxo�I  
�          @�Q���(�@z=q>���@hQ�CL���(�@c�
?�33Atz�Cٚ                                    Bxo�W�  
�          @������@y��=#�
>�{C�����@j�H?��A@Q�C8R                                    Bxo�fZ  T          @�G����H@|(���(��~{C�����H@w�?^�RA z�CY�                                    Bxo�u   
�          @�G���(�@�
=>��R@3�
C	����(�@x��?�z�AuC��                                    Bxo���  T          @�{��\)@~�R>\)?�ffC���\)@n{?�AT��C�
                                    Bxo��L  "          @\��z�@w
=�333��(�C:���z�@xQ�?
=@���C
                                    Bxo���  
�          @�G�����@x�ÿG���C@ ����@|(�?�@�p�C
�                                    Bxo���  
�          @������@i���������C�)����@�G��
=q����C��                                    Bxo��>  "          @����Q�@h�ÿ�33�YG�C���Q�@x�þ����p�C33                                    Bxo���  	�          @�G���\)@HQ쿯\)�Q�C�{��\)@Z=q��z��1G�C�H                                    Bxo�ۊ  �          @������
@`�׿�{�S�CǮ���
@p�׾.{�˅Cٚ                                    Bxo��0  �          @����ff@L���(Q���C� ��ff@w���(��fffC	\)                                    Bxo���  �          @�
=���@Z�H��Q���p�C�����@w
=�=p���(�CQ�                                    Bxo�|  "          @��R��33@g
=������C@ ��33@��ÿ#�
��z�C	+�                                    Bxo�"  �          @�
=����@\(������{C�\����@u��(����\C�=                                    Bxo�$�  T          @�����@j�H��z����C8R���@�Q����z=qC	��                                    Bxo�3n  �          @�  ��ff@p  ����K�C�f��ff@~{�u�
=qC
B�                                    Bxo�B  
�          @�����{@Z=q�Ǯ�qp�C�3��{@n�R�����uCz�                                    Bxo�P�  
�          @�����
=@g
=�У��}�C���
=@|(�����z=qC
�{                                    Bxo�_`  �          @�33���@\)��\)�33C�����@8Q�@  ���HC��                                    Bxo�n  �          @�\)�c33@�G���{�yp�C ��c33@�=q���Ϳh��B�Ǯ                                    Bxo�|�  �          @�����=q@�33��(���ffCǮ��=q@���{�O\)C�\                                    Bxo��R  �          @���|(�@�G���G����HCL��|(�@�(�����|��C�                                    Bxo���  
�          @�=q���
@r�\��  ��
C�����
@z=q>u@G�C�q                                    Bxo���  "          @�����\)@u��33���C	�)��\)@���}p��Q�C(�                                    Bxo��D  �          @�����@r�\�'���  C����@�p���ff�D(�C��                                    Bxo���  
�          @��
����@c33�9����
=C
�����@��ÿ�33�z�\C5�                                    Bxo�Ԑ  �          @��R�w�@q��   ��(�C�R�w�@��
�����9p�C��                                    Bxo��6  
�          @�\)�j�H@g
=��R��\)Cz��j�H@�ff��p��G
=C(�                                    Bxo���  
�          @����X��@xQ��*�H����C(��X��@�Q쿪=q�Tz�B���                                    Bxo� �  
�          @�Q��U�@tz��.�R��z�C)�U�@�\)��z��c�B�W
                                    Bxo�(  "          @����o\)@\)��p�����C(��o\)@��\��
=��p�C��                                    Bxo��  
�          @�ff�c33@y���������CQ��c33@�G��z���
=C�H                                    Bxo�,t  �          @�33�k�@�Q쿑��=C���k�@�p�>��?\Cs3                                    Bxo�;  T          @��N�R@\����
��ffC#��N�R@\)����H  C �q                                    Bxo�I�  �          @�{���\@������z=qB�33���\@Z=q�j=q�7Q�B�33                                    Bxo�Xf  T          @�
=�
=@z�H��{���HB�(��
=@��������
B�\                                    Bxo�g  �          @�Q��  @N{�AG��G�B����  @\)��33���B��                                    Bxo�u�  �          @�G��Mp�@<�Ϳ����C	\)�Mp�@XQ�Y���%�C��                                    Bxo��X  !          @����Dz�@p�׿s33�1p�C33�Dz�@w�>u@1�C c�                                    Bxo���  
�          @��R�AG�@l�ͿG��G�C8R�AG�@p��>\@�\)C �q                                    Bxo���  "          @��
�p��@Vff?�@�33C
J=�p��@A�?�G�A��RC�                                    Bxo��J  "          @������@O\)?���AB=qC\���@/\)@�
A��C��                                    Bxo���  �          @�\)���@J�H?�G�AT(�C�����@(��@
=qA�33C}q                                    Bxo�͖  
�          @���}p�@n{?���A^�RC�{�}p�@H��@=qAˮC�H                                    Bxo��<  
Y          @����}p�@g
=?�AE�C	���}p�@E@��A���C�                                    Bxo���  #          @�G��xQ�@hQ�?�
=ApQ�C�R�xQ�@A�@��Aә�C\                                    Bxo���  
�          @���xQ�@c�
?�33Am��C	z��xQ�@>{@��A��C��                                    Bxo�.  �          @��q�@�
=>\@r�\C���q�@z=q?�\)A���C\                                    Bxo��  
�          @��
�u�@���?333@�p�Cn�u�@i��?��A��\CaH                                    Bxo�%z  �          @��R�|��@�Q�?h��A�HC�
�|��@c33@z�A�  C

=                                    Bxo�4   T          @�(��p  @�33?fffA\)CxR�p  @h��@�A��C�
                                    Bxo�B�  "          @�p��e�@\)?O\)A
=qC�3�e�@c�
?�(�A�  C(�                                    Bxo�Ql  
�          @��
�c�
@|��?B�\AffC\�c�
@b�\?�33A�C(�                                    Bxo�`  T          @����`  @y��?Tz�Az�C���`  @^{?���A���C+�                                    Bxo�n�  "          @�G��q�@�33?�@��C�R�q�@~�R?�=qA��
CxR                                    Bxo�}^  �          @�(��u�@�?\)@�
=C���u�@�=q?���A�ffC=q                                    Bxo��  �          @��z�H@��?#�
@ƸRC��z�H@���?�z�A��\CL�                                    Bxo���  �          @���vff@��\?:�H@�RC���vff@z�H?�p�A�Q�C�                                     Bxo��P  �          @�ff�mp�@��H?:�H@��C �f�mp�@�@33A�
=C�
                                    Bxo���  "          @�����(�@mp�?�
=A9p�C�R��(�@L��@{A�C�{                                    Bxo�Ɯ  T          @�G����@U?��HA9�C�=���@5�@Q�A�{C�                                     Bxo��B  �          @�����\@fff?E�@�C�����\@N{?��A�{C�                                    Bxo���  T          @�Q��s�
@\)�n{�8��C���s�
@*=q�#�
���RC#�                                    Bxo��  �          @�
=�7
=����\)�_z�C6xR�7
=?�\)���\�T{C�3                                    Bxo�4  T          @�p����R@+�>�p�@���C����R@��?���AH��C0�                                    Bxo��  
�          @�ff�]p�@X��@��A���C�{�]p�@"�\@Q�B��C��                                    Bxo��  �          @��
�@��@3�
A���BꞸ�@Dz�@w�B8�B��                                    Bxo�-&  
Z          @��׿���@w
=@G
=B��B�����@2�\@��HBO\)B                                    Bxo�;�  �          @�����@>{@4z�B#�B�p����@�@c�
B_�B��\                                    Bxo�Jr  �          @���Q�>��H?���A���C+xR�Q�>#�
?�A���C133                                    Bxo�Y  �          @�33���\��\)�L�Ϳ�CC\���\�����G�����CBL�                                    Bxo�g�  "          @�
=��Q�aG������eC6n��Q���\��z�C5J=                                    Bxo�vd  �          @�����{�L��?�R@�ffC6���{��p�?�@�=qC7޸                                    Bxo��
  �          @�  ��{?(�>W
=@
=qC-�f��{?�>�p�@x��C.��                                    Bxo���  T          @�p�����Ǯ>��R@^{CE�{����˅����ٙ�CF�                                    Bxo��V  �          @�p���  ���\����Q�C@)��  ��ff��
����C9ff                                    Bxo���  �          @�ff��33?ٙ�@�
Aə�C����33?xQ�@,��A��HC(
=                                    Bxo���  "          @����_\)@<(�@{A�G�C��_\)@�@=p�B�C                                      Bxo��H  �          @�=q�|��?�=q@^{B�C33�|��?B�\@uB.�RC){                                    Bxo���  �          @�����H?�@Tz�B�
C$޸���H>8Q�@aG�B
=C1�H                                    Bxo��  �          @�����33?�\@G�BG�C�\��33?Q�@_\)B�C)aH                                    Bxo��:  T          @�����?�(�@C�
B�\Cn���?��\@`  B33C&�                                    Bxo��  "          @�{�g
=@^{@�A��C(��g
=@.�R@AG�BC�3                                    Bxo��  �          @�z��E@~�R?��A�p�B����E@R�\@;�B��C8R                                    Bxo�&,  �          @����^�R@vff?�(�A���C��^�R@Mp�@/\)A�z�C	J=                                    Bxo�4�  �          @����
=@.{?���Ao33C���
=@��@
=qA�Q�C�                                    Bxo�Cx  T          @�ff��G�@-p�?�Ag�
C���G�@��@Q�A�CY�                                    Bxo�R  T          @���|��@E?�ffAh��C��|��@&ff@�A���C��                                    Bxo�`�  �          @�{�@  @���?�ffA2�\B��{�@  @{�@\)A�33B��q                                    Bxo�oj  "          @���S33@���?J=qA  B�
=�S33@z=q?��HA���C33                                    Bxo�~  T          @�Q��W�@�{?��\A+�C ���W�@o\)@	��A�
=C�                                    Bxo���  
�          @�\)�E@��?�  AS�B�B��E@r�\@��A��C+�                                    Bxo��\  �          @����e@�z�?#�
@�z�C  �e@r�\?�\A�G�Cu�                                    Bxo��  �          @�  �p  @u�?uA#33CT{�p  @Z=q?��RA�C	�f                                    Bxo���  �          @�ff�p��@qG�?���A5�C��p��@Tz�@z�A�\)C
�                                    Bxo��N  "          @�{�dz�@u?���AbffC��dz�@U�@A̸RC	�                                    Bxo���  T          @�\)�`  @u?�33A�p�CQ��`  @N�R@)��A��C	B�                                    Bxo��  �          @����qG�@'
=@>�RB
=CE�qG�?�
=@e�B#�HC��                                    Bxo��@  �          @���h��@hQ�@  A�\)C{�h��@7�@J�HB	z�CǮ                                    Bxo��  "          @���g
=@n{@��A�
=C!H�g
=@?\)@E�B33Cc�                                    Bxo��  "          @��
�u@e?�(�A�(�C���u@:=q@8Q�A�
=C�\                                    Bxo�2  �          @�{��@XQ�?�{A��RC  ��@/\)@.{A�z�C��                                    Bxo�-�  T          @���q�@�z�?��A8  Cn�q�@j�H@�RA�C�f                                    Bxo�<~  �          @����C33@u@@  A���C c��C33@7�@|(�B,��C��                                    Bxo�K$  
�          @�33�N�R@I��@g�B{C���N�R@�\@�33BB��C                                    Bxo�Y�  T          @�{�e@R�\@XQ�B{C	s3�e@\)@���B1�
C��                                    Bxo�hp  
�          @�  �g
=@g�@HQ�A�C���g
=@(Q�@�  B'ffC�                                    Bxo�w  "          @�\)�L��@fff@E�B=qC�H�L��@(Q�@|(�B.ffC��                                    Bxo���  "          @��H�L(�@e@6ffA�p�C�f�L(�@+�@n{B'  C�                                    Bxo��b  T          @����N�R@q�@%�AۅCu��N�R@<(�@`��BQ�C	��                                    Bxo��  �          @�����{@1�@333A�RCz���{?�z�@\��B33CxR                                    Bxo���  
(          @�z῏\)@��R>�
=@�
=B�Q쿏\)@���?�A���B���                                    Bxo��T  "          @�Q�?��@��fff�&{B���?��@�  >���@��B�{                                    Bxo���  �          @��?��
@����Y���33B�8R?��
@��H?
=q@��\B�z�                                    Bxo�ݠ  
�          @���?�@�\)��Q��K�B�  ?�@��
>L��@
=qB��f                                    Bxo��F  �          @��@��@�녿����j�HBx��@��@�  ��\)�B�\B|�R                                    Bxo���  T          @��
@   @�\)��33�s�Bsz�@   @�{�\)���RBw�                                    Bxo�	�  
�          @��
@@��@{���p���(�BR(�@@��@�=q�Y���ffB\p�                                    Bxo�8  U          @��@X��@XQ��(����B3@X��@z=q��Q��x��BDff                                    Bxo�&�  �          @�Q�@h��@Tz��(���Q�B)��@h��@j=q�E���
B4��                                    Bxo�5�  !          @��@�z�@	���Q�����AУ�@�z�@(�ÿ�p����HA�Q�                                    Bxo�D*  
�          @��H@�ff?�  �=q��33A�33@�ff@������A�\)                                    Bxo�R�  �          @��
?���@U��aG��i��B��=?���@QG�?+�A0��B��                                    Bxo�av  T          @�=q��  @�?˅A�z�B�33��  @�=q@:=qA�Q�BҮ                                    Bxo�p  �          @�ff����@���?��A9p�B��׾���@���@��A��B��                                     Bxo�~�  T          @��>aG�@�33?0��@��HB�ff>aG�@�
=@A�  B�
=                                    Bxo��h  
�          @�Q�>.{@�\)�����\B�� >.{@�=q?��Ab=qB�ff                                    Bxo��  "          @�\)��p�@�\)?�@��HBϳ3��p�@���?�\)A�{Bє{                                    Bxo���  �          @���{@�
=?�z�A�\)B���{@���@EBffB߀                                     Bxo��Z  �          @�����@�
=?�@���B�G���@���?���A���B�\                                    Bxo��   T          @��R��\)@�  ?��
A/33B����\)@���@z�A�B���                                    Bxo�֦  �          @�����z�@�z�?(��@�(�B��Ϳ�z�@���@   A�p�Bߏ\                                    Bxo��L  �          @��׿�p�@��?c�
A��B��
��p�@�z�@��A�{B�G�                                    Bxo���  �          @�z��i��@j=q@�A��\C�H�i��@>�R@@��Bz�C��                                    Bxo��  T          @���dz�@qG�?�Q�A�{C}q�dz�@G�@7
=A���C
�
                                    Bxo�>  T          @��H�W
=@qG�@�A���C� �W
=@AG�@QG�B�RC
!H                                    Bxo��  �          @�z��;�@��@{A�z�B�Q��;�@`��@P��B�
C�H                                    Bxo�.�  �          @�33���\@%@#33A�Q�C�=���\?�@H��B33C�                                    Bxo�=0  �          @�(�����?�@-p�A�ffC&f����?��@G�B��C%�                                    Bxo�K�  �          @�(���{?���@   A��Cٚ��{?���@8��A�33C&��                                    Bxo�Z|  �          @�p���(�@ff@,��A�(�C����(�?Ǯ@N{B
��C aH                                    Bxo�i"  T          @����H@P��@G�A�G�Ck����H@#�
@B�\B ��C�R                                    Bxo�w�  "          @�ff�p��@Z�H@#�
A�C	���p��@(��@W
=B�HC�                                    Bxo��n  �          @�ff����@hQ�?��HA��
C	�����@C�
@&ffA�33C�q                                    Bxo��  �          @�{�p  @z=q?�A��C�p  @U@(Q�A�33C
@                                     Bxo���  �          @���n{@z=q?�33A��
C���n{@Vff@'
=A�(�C	�q                                    Bxo��`  �          @�
=�l(�@���?��AQp�C���l(�@j�H@z�A�ffC&f                                    Bxo��  �          @�{�S33@�=q>�Q�@l(�B����S33@�=q?��A|z�B��R                                    Bxo�Ϭ  �          @�{�n{@�\)?��@\C^��n{@{�?�A��Cu�                                    Bxo��R  �          @������@e�?\As�C�����@Dz�@��AîCY�                                    Bxo���  "          @������@y��?�
=A:�RC������@^{@��A�33C�                                    Bxo���  �          @�G��p��@���?��HAjffC.�p��@hQ�@�RA�\)C                                      Bxo�
D  �          @�  �?\)@�  ?��\AK�
B�\)�?\)@�Q�@��Aə�B�{                                    Bxo��  �          @����N{@��\?��RA�=qB�G��N{@j�H@B�\A�\)CE                                    Bxo�'�  T          @�ff�N{@�G�?�A�p�B�Ǯ�N{@i��@=p�A�=qCff                                    Bxo�66  T          @�Q��#33@�ff?��RAP��B����#33@�
=@=qA�  B�B�                                    Bxo�D�  T          @���
=@�  ?�\)AhQ�B��f�
=@��@#33Aݙ�B�G�                                    Bxo�S�  �          @����\)@���?��A?�B���\)@�=q@�A�(�B��                                    Bxo�b(  �          @�\)=�@�z�O\)�z�B�u�=�@�p�?��@��B�z�                                    Bxo�p�  T          @�Q�z�H@�(���=q�7
=BȔ{�z�H@���?�{A;�B�                                      Bxo�t  "          @�\)��=q@����!G���p�B�
=��=q@���?=p�@��HB��                                    Bxo��  �          @�\)��Q�@��ÿJ=q���B�\)��Q�@���?z�@�(�B�=q                                    Bxo���  �          @�  ���@����h����
B�B����@��>�@�
=B�                                    Bxo��f  T          @��׿��@�
=�����9�Bг3���@��\>��@0  B�#�                                    Bxo��  �          @��\��
=@�>���@���B��H��
=@��?��HA��B��                                    Bxo�Ȳ  �          @��\@���?5@��RB՞��\@�{?�(�A�=qB��                                    Bxo��X  �          @�녿p��@���B�\��BȔ{�p��@�?z�@ȣ�B�z�                                    Bxo���  "          @�33?��H@�  ���{B��f?��H@�z�^�R��HB�=q                                    Bxo���  �          @�ff@Fff@fff�ff��ffBE(�@Fff@��׿�{�E�BQ\)                                    Bxo�J  �          @��R@9��@}p�������z�BW  @9��@���
=q���B^Q�                                    Bxo��  "          @�
=@ ��@��Ϳ޸R��{B��@ ��@�{���ʏ\B��3                                    Bxo� �  T          @���?��H@�Q쿅��:ffB�k�?��H@��
>L��@
=qB�=q                                    Bxo�/<  �          @��\?���@�>�Q�@|(�B��\?���@�p�?�z�A�(�B�\)                                    Bxo�=�  T          @�z�?��@��R>���@I��B�=q?��@��R?���A�z�B��H                                    Bxo�L�  T          @�?���@�>�z�@E�B���?���@�?˅A�=qB�                                    Bxo�[.  T          @��R?��@�{>.{?�ffB���?��@�\)?�(�A{\)B��\                                    Bxo�i�  �          @���@ff@��H���Ϳ���B��@ff@�ff?�
=AF�\B���                                    Bxo�xz  "          @�Q�@��@��þǮ��=qB�Ǯ@��@��R?c�
A��B�=q                                    Bxo��   �          @���?ٙ�@��<�>��
B�
=?ٙ�@�=q?���Ab�RB��H                                    Bxo���  T          @���?���@�p�?z�HA$z�B��?���@�Q�@  A��B��q                                    Bxo��l  T          @�=q?�G�@�
=?�33A?�
B���?�G�@�Q�@�A�\)B��R                                    Bxo��  �          @��?�R@����E��
�HB���?�R@�G������ffB���                                    Bxo���  "          @��>�(�@���X����
B�>�(�@��\�(����B�Q�                                    Bxo��^  �          @��>�z�@�z��8Q���(�B�Ǯ>�z�@�
=�У���=qB��{                                    Bxo��  
�          @��>��@�33��=q����B��{>��@��Ϳ   ��  B�#�                                    Bxo���  �          @��׽�G�@��
�����H��B�k���G�@�  >B�\?���B�\)                                    Bxo��P  T          @��׾�@�=q��=q�_�
B�=q��@��=#�
>�(�B��                                    Bxo�
�  �          @�  �W
=@���h���Q�B�ff�W
=@��R>��@�{B�\)                                    Bxo��  �          @��?E�@�33�\(��G�B��{?E�@���?   @�\)B��R                                    Bxo�(B  
�          @�  ?��@�ff�����`z�B�p�?��@�(��L�;�B��q                                    Bxo�6�  �          @���?���@�
=��z��l��B�8R?���@�p������B��3                                    Bxo�E�  T          @��@
�H@���p��MB�\)@
�H@��\=#�
>ǮB���                                    Bxo�T4  �          @��H@"�\@��H�����6{Byff@"�\@�
=>\)?�Q�B{�R                                    Bxo�b�  �          @�33@AG�@��\��  ��33B\z�@AG�@�(���R��Bc��                                    Bxo�q�  �          @���@G
=@s33�2�\��
=BJ��@G
=@��
��G���33BZ\)                                    Bxo��&  �          @��@&ff@aG��X���
=BV33@&ff@�Q��=q��Q�Bj=q                                    Bxo���  �          @���@@o\)�Q��ffBg��@@�{�\)���HBx�
                                    Bxo��r  T          @���@(��@g��P  ��
BWp�@(��@�=q�  ���Bj                                      Bxo��  �          @�p�@'�@x���)����B`33@'�@�p�������Q�Bm�                                    Bxo���  �          @���@33@����,����Bq{@33@�=q��\)��=qB}(�                                    Bxo��d  T          @�\)@2�\@l���8�����BS@2�\@��������G�Bc��                                    Bxo��
  �          @�z�@   @w��-p����Bdz�@   @�p�������Br                                      Bxo��  "          @���@���@ �׿�\)����A�33@���@9����Q��N�RBG�                                    Bxo��V  �          @���@J=q@A��&ff����B/�@J=q@e���\����BB33                                    Bxo��  �          @��@��@r�\�>�R�\)Bf�@��@��Ϳ������
Bu�                                    Bxo��  �          @�Q�?L��@�\)�.�R�\)B���?L��@��ÿ�����  B�
=                                    Bxo�!H  T          @�Q�?˅@XQ��P���$z�B���?˅@��H���߮B�k�                                    Bxo�/�  �          @��\?�=q@6ff�}p��J{Bs�
?�=q@o\)�HQ���\B��                                    Bxo�>�  
�          @�p�?��?�Q��n�R�Z�HB=��?��@333�HQ��-G�Be��                                    Bxo�M:  �          @��
?��R@\)���=qB�u�?��R@�{����y�B���                                    Bxo�[�  "          @��;�z�@�(�����=qB��\��z�@�{�5�z�B�\                                    Bxo�j�  T          @�{�#�
@�G��5��B�=q�#�
@�=q>��H@���B�(�                                    Bxo�y,  �          @��H��  @���?   @��B��H��  @���?�{A��\B�\)                                    Bxo���  �          @���>\)@�33����P  B�ff>\)@�
==���?�G�B��                                     Bxo��x  �          @�(�>k�@vff�!��33B�B�>k�@����G���z�B�
=                                    Bxo��  T          @�p�����@h�ÿ�����HB�Lͽ���@\)�n{�Q�B�                                      Bxo���  �          @�p���{?�{?�A�z�C#���{?fff?��HA��RC(�                                     Bxo��j  �          @�Q���
=@ ��?�33AK
=C�R��
=?�z�?�{A���C �)                                    Bxo��  T          @��
��
=@�
?G�A��C����
=@�\?���Ap��Ch�                                    Bxo�߶  �          @������H>���?�(�AW
=C/�=���H=�Q�?��
Aa�C3
=                                    Bxo��\  �          @��
��ff?�
=?�33AE�C!B���ff?���?\A�ffC$�=                                    Bxo��  �          @���  ?�G�?�Q�AJ�HC ����  ?�?˅A�{C$.                                    Bxo��  "          @�����  ?�z�?�{Ac\)C\��  ?\?��
A���C#{                                    Bxo�N  �          @������@��?}p�A)C����?���?\A��RC�f                                    Bxo�(�  �          @�  ��=q@p�?�(�Aw\)Cff��=q?��
?��HA�G�C�q                                    Bxo�7�  T          @����@{?�(�A}��C����?��?�(�A��C
                                    Bxo�F@  �          @�
=��Q�@#�
?z�HA�C���Q�@\)?�=qA�=qC�H                                    Bxo�T�  �          @�����{@Q�?L��@�
=CW
��{@
=?�{AZ{C�
                                    Bxo�c�  �          @�G���  @�?:�H@���C\��  @�?��
AL��CaH                                    Bxo�r2  �          @�ff����@�>�@�p�C@ ����@(�?��
A'
=C��                                    Bxo���  �          @���(�@�H>�(�@��
C���(�@\)?�G�A$(�Cn                                    Bxo��~  T          @�z���(�@E��  �)��C�{��(�@C�
?�@���C�                                    Bxo��$  �          @�33��@:=q�:�H���C����@@      <�C��                                    Bxo���  T          @�����@%�G��ffC����@,�ͽ����C�H                                    Bxo��p  �          @�33���@*=q�\)����C���@-p�=�G�?�33C}q                                    Bxo��  �          @��H����@E�z���\)CG�����@H��>L��@
=C޸                                    Bxo�ؼ  �          @��\��@I���z�H�*{C���@S33�8Q���HC�                                    Bxo��b  
�          @�����@:=q����4Q�C�q���@Dzᾔz��J�HC��                                    Bxo��  �          @�=q��33@5������D��C����33@AG�������33C(�                                    Bxo��  �          @�Q��y��@H�ÿ\��
=C#��y��@Z�H�333��\)C
                                    Bxo�T  T          @����z=q@I���Ǯ��(�C5��z=q@[��=p��{C
�                                     Bxo�!�  �          @���u�@Dz��   ��
=C:��u�@^{��
=�N�\C	�\                                    Bxo�0�  �          @�=q�}p�@(���Q���=qC@ �}p�@H�ÿ�z���33C�                                    Bxo�?F  T          @��H�~{@3�
�(���=qC���~{@P�׿�Q��z�\C��                                    Bxo�M�  �          @����qG�@7���\��(�CǮ�qG�@U�\��{C
�                                     Bxo�\�  �          @�Q��XQ�@X�������C�H�XQ�@s33��Q��R{C��                                    Bxo�k8  T          @���N�R@9���/\)�
=C
��N�R@^�R������=qC��                                    Bxo�y�  T          @���Dz�@_\)�\)�ͮCO\�Dz�@|(������lz�B��                                    Bxo���  �          @��
�=p�@Fff�/\)�p�C�=p�@j�H��33����C ��                                    Bxo��*  �          @�(��,��@<(��<(��z�C���,��@e��Q��ˮB�(�                                    Bxo���  �          @�z��*=q@b�\�k��<Q�B��)�*=q@j=q����p�B�\                                    Bxo��v  T          @�p���G�@AG�?��A=��C8R��G�@*�H?�G�A�(�C�=                                   Bxo��  �          @������@Fff?\)@ə�Cc�����@7�?�ffAlQ�C�                                   Bxo���  T          @�  @�@
�H�xQ��G�B'
=@�@C�
�N�R�=qBM�
                                    Bxo��h  �          @��H@�?����R�c�\B{@�@6ff�xQ��;33BMz�                                    Bxo��  T          @��H?�=q@z����a=qBOz�?�=q@U�n�R�1�Bu\)                                    Bxo���  �          @�=q?У�@�\��
=�gp�BZQ�?У�@S�
�qG��6�\B\)                                    Bxo�Z  �          @�Q�?��@.�R�����[�\B���?��@l(��`  �'�B���                                    Bxo�   �          @�
=?У�@����\�dG�BY�
?У�@QG��h���3z�B~{                                    Bxo�)�  T          @�ff?��H@(���=q�e{BPz�?��H@L(��j=q�5Q�BwG�                                    Bxo�8L  �          @�?�Q�?��R����e��B7
=?�Q�@@  �o\)�9Q�Bdp�                                    Bxo�F�  T          @�ff?�?�33��ffaHBE{?�@0  �����U\)Bz�
                                    Bxo�U�  �          @���?��H?s33��G�(�A�(�?��H@����d
=B;p�                                    Bxo�d>  �          @�ff?Ǯ?�33���W
B�
?Ǯ@p���=q�ap�B[�\                                    Bxo�r�  �          @��>aG�@E�����Q��B���>aG�@~�R�I���33B�Ǯ                                    Bxo���  �          @��׼#�
@a��_\)�2�
B�ff�#�
@����!G���(�B�Q�                                    Bxo��0  �          @�ff?�?��H�U��jG�B�u�?�@-p��0  �3(�B�=q                                    Bxo���  �          @���J=q@[��l���9�B��)�J=q@���0  �{B�{                                    Bxo��|  �          @�\)���@�p������p�B�����@�33��z��L��B�aH                                    Bxo��"  �          @�Q��(Q�@e��Mp���\B��\�(Q�@�Q��\)����B�W
                                    Bxo���  �          @��ÿ�ff@��C�
���B���ff@�����Q����B�
=                                    Bxo��n  �          @�����@�p��'
=�ۮB�z����@�p������b{BڸR                                    Bxo��  �          @�(���
=@���L���
��Bހ ��
=@�
=��\��Q�B�Q�                                    Bxo���  
�          @�(���  @�Q��S33���B�Ǯ��  @�{�	����
=B�                                    Bxo�`  �          @��� ��@�=q�#33��B���� ��@�녿�z��r�RB�aH                                    Bxo�  �          @��H�   @���Ǯ���B�p��   @��;�p��vffB�{                                    Bxo�"�  �          @�Q��B�\@��R?8Q�@�33B�z��B�\@���?��A��B�k�                                    Bxo�1R  T          @�G��1�@��H���H�K\)B��1�@���#�
���B���                                    Bxo�?�  �          @����.�R@����=q��  B�z��.�R@�
=�0����Q�B���                                    Bxo�N�  T          @����(�@�z��&ff�ߙ�B����(�@�zῸQ��p��B�#�                                    Bxo�]D  �          @��
�@�G��G��  B�(��@������B�{                                    Bxo�k�  T          @��R��
=@z=q�j=q�   B�׿�
=@�{�%���  B���                                    Bxo�z�  �          @�Q��  @�{�N�R��B� �  @���ff���B�q                                    Bxo��6  �          @�Q쿬��@s�
�����3ffB�  ����@�{�=p���  B�#�                                    Bxo���  �          @�p��B�\@`  ���R�S
=B̞��B�\@����l����B�
=                                    Bxo���  �          @��Ϳ��@N{��p��a�\Bǅ���@��\�~{�)
=B\                                    Bxo��(  �          @��
�G�@9�������l�
B�#׿G�@����p��5{B���                                    Bxo���  �          @��׾k�?�(����
8RB�B��k�@?\)�����g�\B���                                    Bxo��t  T          @�\)?fff@P  ��{�^�B�.?fff@��
�\)�'=qB��
                                    Bxo��  T          @��H?E�@6ff��G��n��B��
?E�@�Q����6��B�Q�                                    Bxo���  �          @�=�G�@^{��
=�JQ�B�k�=�G�@����N�R�{B�.                                    Bxo��f  �          @��\�Tz�@�Q��g��%��B�W
�Tz�@����!G���G�Bǣ�                                    Bxo�  �          @�ff�+�@���������(�B�  �+�@�z�B�\��G�B�p�                                    Bxo��  �          @�  �I��@��
�+����HB��=�I��@���>�(�@��\B�.                                    Bxo�*X  T          @�ff�^�R@|��?��AdQ�Cc��^�R@_\)@G�A�(�C��                                    Bxo�8�  
�          @���\(�@{�@�A���C5��\(�@Q�@A�B�C\)                                    Bxo�G�  �          @�G��S�
@�  ?�Q�AO
=C���S�
@e�@	��A�=qC�                                     Bxo�VJ  �          @�\)�#�
@�
=?��@��HBꞸ�#�
@�?�G�A�ffB�z�                                    Bxo�d�  
�          @���:=q@�G��O\)�	G�B�B��:=q@�33>�{@dz�B��{                                    Bxo�s�  �          @��\����@��R�G���p�C�\����@���>�\)@1�CaH                                    Bxo��<            @�G����R@����z��1G�CW
���R@�=q��Q�W
=CE                                    Bxo���  �          @�G����@g
=��  ����C
�q���@|(��L�����C�\                                    Bxo���  �          @�=q��=q@s33�&ff����C
����=q@��\�����g
=C�f                                    Bxo��.  �          @��\�u@N{�>�R���C�R�u@w�����p�C��                                    Bxo���  �          @��
�?\)@�{�J=q����B�z��?\)@�\)>�@��\B�\                                    Bxo��z  �          @��R�Q�@�������B��H�Q�@��\?E�@��
B�(�                                    Bxo��   �          @���   @�33�\)��z�B�.�   @�=q?:�H@�33B�\)                                    Bxo���  �          @�=q��R@�33�Ǯ�y�B�L���R@�=q�k���
B�                                     Bxo��l  �          @��\�)��@��׿�����B���)��@��\�����RB�#�                                    Bxo�  �          @���J=q@��\�#�
���
B�W
�J=q@��\?z�@���B�B�                                    Bxo��  �          @�(��8Q�@����ff��p�B�z��8Q�@�G������p�B�k�                                    Bxo�#^  �          @��\�>{@����\)�\��B�\�>{@��\)��B�                                    Bxo�2  �          @��H�S�
@�Q��
�H��\)B���S�
@�p����
�#�B���                                    Bxo�@�  �          @��H�/\)@�G��G���33B�.�/\)@�
=����(��B�                                    Bxo�OP  �          @�(�� ��@�G��ff����B�p�� ��@��ͿE����
Bܽq                                    Bxo�]�  �          @����.{@��������RB�#��.{@�녿�z��6ffB�=                                    Bxo�l�  �          @���N{@����G���B��H�N{@��R��\)�1B���                                    Bxo�{B  �          @��R�Y��@��H�$z���=qC���Y��@�����H�dz�B���                                    Bxo���  �          @�
=�-p�@�{�8Q��癚B����-p�@��ÿ���=qB�R                                    Bxo���  �          @��R�޸R@���`�����B�LͿ޸R@�\)�G���33Bؽq                                    Bxo��4  �          @�{�G�@i����Q��+�B�q�G�@�G��<����B�{                                    Bxo���  �          @������@@����p��C��C \)���@����`����B�8R                                    Bxo�Ā  �          @���  @��\�n�R�z�B�LͿ�  @�(�� ���ˮB�u�                                    Bxo��&  T          @��׾�ff@�Q���\)�2�B����ff@�ff�AG���ffB��                                    Bxo���  �          @�녽�G�@�33��{�=33B�.��G�@��H�P����\B��\                                    Bxo��r  �          @\�\@qG���
=�M  B��׾\@��H�g
=�z�B�\                                    Bxo��  �          @\���@\�����R�\�B�����@�33�z�H�!��B��q                                    Bxo��  �          @��;\)@G
=��G��n
=B�.�\)@�33��=q�3=qB��3                                    Bxo�d  �          @����@[����\�^��B��Ϳ�@�33��G��$G�B�
=                                    Bxo�+
  �          @�{�B�\@*=q���\�HB��B�\@\)��
=�F�B�\)                                    Bxo�9�  �          @�ff����@g���  �X{B�k�����@����z�H�G�B��                                    Bxo�HV  �          @�{��\@�z������=ffB���\@���Tz���B�33                                    Bxo�V�  
�          @��þ\)@��������B�ff�\)@�p��Tz���B�#�                                    Bxo�e�  �          @����\@�Q�?�\)AzffB�33��\@��@B�\A�
=B�ff                                    Bxo�tH  �          @�=u@�G���{�-G�B��R=u@�(�>��@~{B��q                                    Bxo���  �          @���?^�R@��ÿ�ff���RB�\?^�R@�녾����L(�B��f                                    Bxo���  �          @��?��H@�z��33����B���?��H@�ff��ff���RB��                                    Bxo��:  �          @���?�ff@��׿��
��(�B�G�?�ff@�����p��j�HB�{                                    Bxo���  �          @��@(�@����p��iG�B��=@(�@�{���
�Q�B�{                                    Bxo���  �          @�G�@C33@��׿����?\)Be��@C33@��=�\)?&ffBh��                                    Bxo��,  �          @��\@S33@��Ϳ����2�HBZ�R@S33@�G�=�G�?�ffB]                                    Bxo���  �          @�=q@�R@�{�.{����B��f@�R@�{?+�@�{B��                                    Bxo��x  �          @�33@P��@��H�������BT\)@P��@�p��333�޸RB\z�                                    Bxo��  �          @�G�@J�H@�
=�c�
��BZ�
@J�H@���>�z�@AG�B\�R                                    Bxo��  �          @\����@�G�>�z�@0��B�Ǯ����@���?��
A�\)B�8R                                    Bxo�j  �          @�33�I��@�{>�z�@0��B�{�I��@�{?�\)Aw�
B�#�                                    Bxo�$  �          @�z��(�@��
?\(�Az�B�W
��(�@��R@{A�ffB�Q�                                    Bxo�2�  �          @��ÿ��R@�  ?��A�G�B�W
���R@��@AG�B �B��                                    Bxo�A\  �          @�  �hQ�@�?�33A��\CT{�hQ�@vff@/\)A��CO\                                    Bxo�P  �          @�=q�vff@�{?�Q�A[�
C�f�vff@z=q@"�\Aƣ�C�                                     Bxo�^�  �          @���  @�ff?�z�A5G�C����  @p  @p�A��
C�\                                    Bxo�mN  �          @Å����@�z�?��@��Cff����@��\?�(�A��Cz�                                    Bxo�{�  �          @�33��=q@�G�=L��?   C޸��=q@��
?�p�A:�RC�R                                    Bxo���  �          @�
=��Q�@���\)����CG���Q�@���?��\A�C�                                    Bxo��@  �          @�33�Q�@�\)��\��\)B�z��Q�@�ff?=p�A ��B���                                    Bxo���  �          @�G���@�ff�0����{B����@�
=?\)@�B�                                     Bxo���  �          @�{�8Q�@�\)�����33B�W
�8Q�@�ff�����B=qB�                                    Bxo��2  �          @�G���=q@�  �(�����B�zῪ=q@��?=p�@��BЊ=                                    Bxo���  �          @��ÿn{@�=q�W
=�z�Bȳ3�n{@��
?�@�{Bȅ                                    Bxo��~  �          @��R�J=q@�
=��z��E�B�33�J=q@�33>�  @,(�B���                                    Bxo��$  �          @�
=���@��H������B��;��@�p������  B�p�                                    Bxo���  �          @��R�aG�@�{�����ffB��=�aG�@������ffB�W
                                    Bxo�p  T          @�  ��G�@��ÿ�z��c�B���G�@��R=���?�G�B�                                    Bxo�  �          @��
>�=q@��ÿ���33B�ff>�=q@��H�����w�B��3                                    Bxo�+�  �          @��H?Tz�@���*=q��\)B�u�?Tz�@�(����H�@z�B�(�                                    Bxo�:b  �          @�=q>�{@�  �;���\B�(�>�{@�33��  �p(�B�                                      Bxo�I  �          @��?�@�
=�2�\��B���?�@��ÿ����^�RB�G�                                    Bxo�W�  �          @�녿�z�@�\)�c�
�+�B��ÿ�z�@��>���@|��B�ff                                    Bxo�fT  �          @�{���
@k�?���Az{C�����
@HQ�@!G�A���Cu�                                    Bxo�t�  �          @��
��(�@Z=q?�A��\C
��(�@3�
@*=qA�(�CY�                                    Bxo���  �          @�
=����@?\)?�z�A�Q�C5�����@�@(��A�C:�                                    Bxo��F  �          @�����@L��@A�(�C
=���@!G�@7�A�C33                                    Bxo���  �          @Å���@`  @33A�\)C�����@4z�@:=qA�(�C@                                     Bxo���  �          @������@'�@&ffȀ\C0�����?�=q@Mp�B ��C                                      Bxo��8  �          @�������?#�
@`��B�C,c��������
@c�
B��C7�
                                    Bxo���  T          @�����
=@B�\?��A��C���
=@�@(��A��
C�q                                    Bxo�ۄ  �          @�{��ff@Dz�@   A���C+���ff@�H@0  A�=qC��                                    Bxo��*  �          @�33��
=@J�H@
�HA�33C���
=@{@<(�A�\)Cc�                                    Bxo���  �          @�Q���(�@N�R@G�A�33C)��(�@$z�@3�
A߮C�q                                    Bxo�v  �          @��H��@Z=q?�33Ag�
C����@:=q@��A��C.                                    Bxo�  �          @����R@K�@
=A��\C�R���R@\)@8��A���Ch�                                    Bxo�$�  �          @�  ����@QG�?�(�A�z�C������@'�@1�A�C��                                    Bxo�3h  �          @�
=��G�@J=q@   A���C����G�@   @1�A�=qC�q                                    Bxo�B  �          @�p���Q�@.{@33A�=qC���Q�@�
@-p�A�G�Cu�                                    Bxo�P�  �          @�{���\@Z�H?���A\  C�����\@:�H@�RA���C�R                                    Bxo�_Z  �          @�녿�{@����������B�aH��{@�z����ffB�.                                    Bxo�n   �          @�33�@�����\)�B{B�\�@�p�>aG�@��B���                                    Bxo�|�  T          @���s�
@���    �#�
C�R�s�
@��?�z�A;
=C�\                                    Bxo��L  �          @�����(�@{�?+�@�  Cc���(�@fff?�(�A���C
��                                    Bxo���  �          @�����@�(�>���@>{C� ���@x��?�AbffC5�                                    Bxo���  �          @��
�}p�@�G�>�G�@���C���}p�@�  ?˅A|��C��                                    Bxo��>  �          @��H��p�@}p�?z�HA{Cz���p�@b�\@G�A�ffC��                                    Bxo���  �          @�
=���@~�R?�{ARffCٚ���@\��@��A�p�C޸                                    Bxo�Ԋ  �          @�Q���Q�@�  ?�33AX  C���Q�@]p�@��A�ffC�H                                    Bxo��0  �          @�\)��ff@s�
?�
=A6�RCh���ff@U@(�A��C)                                    Bxo���  �          @�  ��\)@|(�?ǮAqC	\��\)@W
=@%A̸RC�
                                    Bxo� |  �          @�(����@qG�?�=qA�
=C	}q���@G�@333A㙚C��                                    Bxo�"  �          @�
=���@~�R?n{A\)C	L����@dz�?��RA�z�Cc�                                    Bxo��  �          @�Q���G�@�(���\)�.{C���G�@���?^�RA�\C�f                                    Bxo�,n  �          @�z��c�
@�\)?
=q@�33C ��c�
@���?�G�A�Q�C�H                                    Bxo�;  T          @�p��q�@��Ϳ����0  C�H�q�@�G�>\)?���C�q                                    Bxo�I�  �          @�z��`  @������
�H��B�33�`  @�\)=#�
>�Q�B��                                    Bxo�X`  �          @�(��S33@������H��G�B����S33@�33��p��i��B��\                                    Bxo�g  �          @�=q�<��@�ff�\)��B���<��@��Ϳh���  B�(�                                    Bxo�u�  �          @�=q�.{@�ff�!���
=B���.{@�\)��Q��<��B�W
                                    Bxo��R  �          @��H�{@�ff�6ff��ffB��{@�=q���R�mB��f                                    Bxo���  �          @����@^�R�~�R�1Q�B�����@��R�6ff���B���                                    Bxo���  �          @�����@~{���R��p�B��f���@��
�L���  B�\)                                    Bxo��D  �          @�{�c33@��?��A%G�B�=q�c33@�(�@�
A�(�C��                                    Bxo���  �          @�
=��Q�@�z�>��@y��Cff��Q�@�33?�\)A}�C\)                                    Bxo�͐  �          @�ff�w�@�
=����ffCٚ�w�@�p�?Q�@�{C.                                    Bxo��6  �          @�33���@G
=���R�Dz�C�H���@E�?z�@�
=C�3                                    Bxo���  �          @������\@g��G���C�H���\@l��>�\)@,(�C{                                   Bxo���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxo�(  	�          @��H�C�
@��H����£�B�L��C�
@��H�����)p�B�                                     Bxo��  �          @���>�R@�{��33��Q�B����>�R@�G��\)����B�                                    Bxo�%t  �          @��\�)��@��ÿ�ff�_\)B��=�)��@�
==#�
>�
=B�{                                    Bxo�4  �          @�p��L��@�  ?�\)A7\)B��
�L��@\)@ffAŮC �3                                    Bxo�B�  �          @���;�@���?�{A�=qB�.�;�@��@9��A�33B�                                      Bxo�Qf  �          @��H�L(�@���?�G�A33B�u��L(�@�G�@�A�z�B�B�                                    Bxo�`  �          @�z��^{@��>�(�@��B�p��^{@�p�?�\A��C &f                                    Bxo�n�  �          @��x��@��?��
A!p�C��x��@qG�@��A��RC�f                                    Bxo�}X  �          @�ff�=q@��
������B�q�=q@�  �&ff��Q�B�u�                                    Bxo���  �          @�\)�0  @���������\B���0  @�
=�J=q����B�\                                    Bxo���  �          @�\)�
=@����Q�����B�{�
=@���Tz�� Q�B�                                    Bxo��J  �          @��R��@�=q��=q�:
=B��=��@����=p���\B��q                                    Bxo���  �          @�{�.{@hQ���{�P�B�p��.{@��\�\(��B�{                                    Bxo�Ɩ  �          @�
=��\)@n{��z��M(�B�{��\)@����W
=�	�
B��{                                    Bxo��<  �          @�z�?#�
@\�����Tp�B�?#�
@���^{���B�aH                                    Bxo���  �          @��?=p�@N�R��=q�]�HB�\?=p�@���j�H��B�33                                    Bxo��  �          @��
?�Q�@A������a��B�{?�Q�@�=q�s33�!G�B�8R                                    Bxo�.  �          @�33?���@aG�����F(�B��
?���@����L(����B�G�                                    Bxo��  �          @�=q?�  @�(��s33�#p�B�.?�  @�����ŮB�\                                    Bxo�z  �          @��?��
@\)��(��4p�B�{?��
@����2�\��=qB��)                                    Bxo�-   �          @�
=?W
=@�{���\�dQ�B�?W
=@��H>k�@"�\B�W
                                    Bxo�;�  �          @���?�33@�33��33��ffB���?�33@�33�#�
���
B��
                                    Bxo�Jl  �          @��>���@��N�R�B�  >���@��������HB��f                                    Bxo�Y  T          @�ff>��H@�
=�e��=qB�=q>��H@������  B�                                    Bxo�g�  �          @�?5@�33�|(��'G�B�ff?5@�=q��R�ƸRB��q                                    Bxo�v^  �          @�z�?W
=@��\����3�B��R?W
=@�(��1G����\B�aH                                    Bxo��  �          @��H?=p�@x����
=�;(�B�u�?=p�@�
=�8Q���ffB�
=                                    Bxo���  �          @�33>k�@�����
=�9Q�B��\>k�@��H�5����B��f                                    Bxo��P  �          @�=q>�p�@����N{���B�(�>�p�@�Q��33��p�B�W
                                    Bxo���  �          @�(�=��
@��a���HB�#�=��
@��׿�(���G�B�k�                                    Bxo���  �          @��
���@��
�z=q�'G�B����@��H�����B�L�                                    Bxo��B  �          @��
�aG�@~�R�����<��B�B��aG�@��\�:=q��z�B��f                                    Bxo���  �          @��H���@�
=��Q��-ffB�\���@�\)�#�
��{B��                                    Bxo��  �          @��H��  @�=q�����/{BϮ��  @��H�'
=���B�L�                                    Bxo��4  �          @��\����@u���p��7  B��쿬��@���5���\B�                                    Bxo��  �          @��H��
=@Q�����O�B�  ��
=@�  �W
=��
B�.                                    Bxo��  �          @��H���@����G��{B�p����@l(���(��C33B�aH                                    Bxo�&&  �          @��\�&ff?˅������B��=�&ff@L(���=q�_�RB�\                                    Bxo�4�  �          @��
�J=q@z���p���B��J=q@g
=�����K��B̨�                                    Bxo�Cr  �          @�zΉ�?������
��C �Ή�@?\)��ff�d��B�33                                    Bxo�R  �          @��
�k�?��
��z��fB�Q�k�@;�����i�RBֽq                                    Bxo�`�  �          @�녿�z�?�\����C@ ��z�@p���\)\B뙚                                    Bxo�od  �          @���
=?�\)��{�q{C
aH�
=@R�\��z��:�B�\)                                    Bxo�~
  �          @�G��333?���z�¡�C�\�333@����ff�B�33                                    Bxo���  �          @��ÿ\(�@���\)�B��\(�@e���33�G�B��                                    Bxo��V  �          @��׿z�H?�z����\#�B�ff�z�H@\������M��B�                                    Bxo���  �          @��׿L��?�����Q���B�Ǯ�L��@Dz������bp�B�(�                                    Bxo���  �          @��ͿG�?���(�k�B�33�G�@@�����aB���                                    Bxo��H  �          @�ff��R?����F�B�=q��R@�Ϳ�{��RB�\)                                    Bxo���  
�          @���  @�
=��z��g\)B��  @��>B�\?���B�#�                                    Bxo��  �          @��\���
@{��C33���B�=q���
@���Q���{BҸR                                    Bxo��:  �          @�{>L��@R�\����Xp�B�33>L��@�G��Tz��z�B�\                                    Bxo��  �          @���>�Q�@e��
=�L(�B���>�Q�@����HQ��G�B�u�                                    Bxo��  �          @�\)>k�@w
=���R�=�HB�Q�>k�@�\)�3�
��\)B���                                    Bxo�,  �          @�Q�.{@������\�5�\B�Ǯ�.{@��H�(Q���ffB�                                    Bxo�-�  �          @��׿0��@��
�n�R�&��B���0��@��\�����ffBÊ=                                    Bxo�<x  �          @�=q����@���}p��.33B�  ����@��{��p�B�8R                                    Bxo�K  �          @��
>�G�@s33��p��Dp�B�Ǯ>�G�@���@����  B��                                    Bxo�Y�  �          @��?(��@mp����R�G(�B���?(��@�p��Dz���{B��
                                    Bxo�hj  �          @��?333@A���Q��h�B���?333@�ff�s�
� �HB��                                    Bxo�w  �          @�p��u@3�
��p��v
=B�W
�u@�G������,��B��=                                    Bxo���  �          @��
�\)@	����p���B��3�\)@p  ��
=�G�B�k�                                    Bxo��\  T          @�G�=u?�����\)Q�B�� =u@\����(��U(�B�(�                                    Bxo��  �          @��?�\)?L�����
�\Bff?�\)@!����\�u�B�u�                                    Bxo���  �          @�G�?��
?�ff��=q��B �H?��
@:=q��(��Zz�Bx�                                    Bxo��N  �          @�
==�Q�@'���G��y�HB�\=�Q�@��H�|���0  B�z�                                    Bxo���  �          @������@p  ����8B��f����@��H�*=q��B��                                    Bxo�ݚ  �          @�G�?(�?fff��=q  B_�H?(�@'�����vB���                                    Bxo��@  �          @��
?�p���  ���H8RC���?�p�?�ff��(�=qB'�                                    Bxo���  �          @��?˅�\��Q��HC�O\?˅?�33���H�B%�                                    Bxo�	�  �          @���W
=@tz῅��9�CaH�W
=@|��>k�@%�Ch�                                    Bxo�2  �          @��R�l��@x��?�R@�(�C�\�l��@_\)?�A�
=C�3                                    Bxo�&�  �          @��\�e�@|(�����=qC5��e�@qG�?�33AG33Cu�                                    Bxo�5~  �          @��H�n�R@r�\��\)�C33C���n�R@k�?s33A$Q�Cff                                    Bxo�D$  �          @�G��s33@hQ�>�\)@EC^��s33@U?�Q�A}��C
�3                                    Bxo�R�  �          @����
=@�H>��@�G�C޸��
=@��?�(�ATz�C�H                                    Bxo�ap  �          @��\�\��@Z=q?��HA`z�C^��\��@5�@  A�C��                                    Bxo�p  �          @�p��K�@N{?z�HAB�RC�\�K�@.�R?�(�A�33CE                                    Bxo�~�  �          @�=q�"�\@(Q�@i��B3�
C��"�\?��@��Ba��C                                    Bxo��b  �          @����7
=@`  @�HA�\)C:��7
=@!�@[�B'C
�\                                    Bxo��  �          @�ff�0��@q�?�(�A��B�L��0��@;�@FffB\)CL�                                    Bxo���  �          @����0  @��
>���@�B�\)�0  @�  ?�ffA��B�                                      Bxo��T  	_          @���aG�@c33?�
=A�
=C�aG�@.�R@?\)Bp�CE                                    Bxo���  �          @�  �`��@���?^�RA\)C!H�`��@`��@Q�A�  C�R                                    Bxo�֠  �          @�=q�p  @~{�u�#33CY��p  @u�?���A4��CaH                                    Bxo��F  �          @�G��u�@E������\)C5��u�@l�Ϳ�  �R�\C�                                    Bxo���  �          @��dz�@}p�?���A^=qC��dz�@S33@#33A��C	=q                                    Bxo��  T          @��\�E�@r�\@=qA�  C
�E�@1�@aG�B!Q�C	�H                                    Bxo�8  
�          @�=q�\)@Z=q@dz�B �
B�Q��\)@�\@��BY�\C��                                    Bxo��  �          @��R�S�
@{�@ffA�Q�C#��S�
@;�@`��B�HC
��                                    Bxo�.�  �          @��b�\@�G�?E�@��C��b�\@s33@	��A��
C                                    Bxo�=*  T          @��/\)@�p�?   @���B�L��/\)@�
=@ffA��RB�                                    Bxo�K�  
�          @���C33@��H=���?�ffB�u��C33@���?��HA�
=B��                                    Bxo�Zv  �          @�{�QG�@�G��ٙ���(�B����QG�@���.{��\B���                                    Bxo�i  "          @�  �ٙ�@�G��333�Q�B�\�ٙ�@������eB�\)                                    Bxo�w�  �          @�Q�u@q��e�.33B��q�u@�Q�����  B�aH                                    Bxo��h  �          @�����=q@`  �qG��7
=B�k���=q@��������B�                                    Bxo��  "          @������@ff��  �w(�BΔ{���@n{�\���*  B���                                    Bxo���  T          @�33��\)@���{�v(�B�LͿ�\)@n�R�i���,��B�u�                                    Bxo��Z  "          @�  ��{?�Q����
�{ffB�
=��{@XQ��l���5�B�Ǯ                                    Bxo��   
(          @�=q�n{?�
=��=q��C c׿n{@4z����
�_G�B؞�                                    Bxo�Ϧ  �          @����xQ�?��������B���xQ�@;���  �X  Bخ                                    Bxo��L  
+          @���?��
��  �d�C�=�@G
=�X���(�\B��                                    Bxo���  
�          @����5?�33�����J�C�)�5@:�H�N{�\)CB�                                    Bxo���  T          @�z��6ff?�\)�w
=�F��Cu��6ff@5��E����C33                                    Bxo�
>  �          @����QG�@2�\�0  �G�C�{�QG�@c�
�У���G�C��                                    Bxo��  %          @�녿�@mp��S33�B��῵@����=q��ffB�L�                                    Bxo�'�  �          @�\)����@s�
�xQ��5�B��f����@����˙�B��3                                    Bxo�60  
�          @��R��@w
=�333� ��B����@��H����f�\B��f                                    Bxo�D�  
�          @���:=q@r�\��33���\B���:=q@�G��#�
�#�
B���                                    Bxo�S|  
�          @�=q�p�@�ff���tQ�B�ff�p�@��>�\)@A�B�{                                    Bxo�b"  "          @��c�
@�(����
�B�\B�Ǯ�c�
@�=q?�  A�33B�                                    Bxo�p�  
�          @�{��
=@�����p��33B��H��
=@�Q�>�=q@7
=B�.                                    Bxo�n  �          @�{��{@S�
?��HA��C����{@�H@>�RA�\)C�                                    Bxo��  %          @�p���=q@P  ?��A��RC���=q@�R@*=qA�RC#�                                    Bxo���            @�G��:=q@�\)?�Q�Ay�B�  �:=q@^{@3�
A���C��                                    Bxo��`  	�          @�����@���?Tz�AG�B�u���@}p�@ffAۅB��q                                    Bxo��  T          @�{�.{@��\?�=qA�B�z��.{@i��@S�
B  B�aH                                    Bxo�Ȭ  "          @����Z�H@��@�
A�=qC��Z�H@@  @e�B�HC
�                                    Bxo��R  "          @�=q�~{@u?���A~{C�f�~{@C33@3�
A��HCk�                                    Bxo���  �          @�G��j�H@Z=q@333A�\)C	��j�H@p�@s�
B&��C�f                                    Bxo���  
�          @�G��/\)@�@��RBQ��C�q�/\)>\@��Br��C,
=                                    Bxo�D  �          @�=q��G�@1G�@>{A��C�=��G�?�ff@p��B#�
C�q                                    Bxo��  �          @�p���ff@g
=?��HAd��C���ff@7�@(��A��C8R                                    Bxo� �  %          @�(����\@XQ�?�z�AeC{���\@*=q@ ��A��
Ck�                                    Bxo�/6  Q          @�(���ff@�G�>��@�G�C)��ff@hQ�?�A��\C!H                                    Bxo�=�  
�          @�=q��ff@�  =�Q�?p��Cn��ff@mp�?�  ApQ�C
�
                                    Bxo�L�  �          @�p��o\)@�z�h����C)�o\)@�{?�R@ə�C�                                     Bxo�[(  �          @�33�n�R@W
=?.{@�
=C	��n�R@9��?�=qA�p�C
                                    Bxo�i�  �          @��\�z�H@U?���AF=qC�
�z�H@.{@\)Aə�CG�                                    Bxo�xt  �          @�  �\)@l(��J=q���C	8R�\)@n�R?z�@���C��                                    Bxo��  
�          @����H@^{�L����
C�3���H@a�>��H@�{C:�                                    Bxo���  %          @�Q����R@Y����  �'33C\���R@a�>�\)@>{C�q                                    Bxo��f  
�          @����33@Vff�=p���{Ch���33@Y��?�\@�Q�C�                                    Bxo��  T          @�=q��{@N�R�   ���RC�R��{@L(�?0��@�RCG�                                    Bxo���  
�          @�  ���@=p��   ��33C�����@^�R�L����\C\                                    Bxo��X  
�          @�G��4z�@s33�.{����B���4z�@��ÿ�z��E�B���                                    Bxo���  
�          @����(Q�@q��8Q��   B���(Q�@�=q�����a�B��                                    Bxo���  �          @�\)�{�@8���  ��C���{�@`�׿�ff�4z�C
J=                                    Bxo��J  T          @�{�1�@~{�ff����B�{�1�@����@  � ��B��f                                    Bxo�
�  
�          @�\)�e�@h�ÿ�33��G�Cz��e�@\)�aG��33C�)                                    Bxo��  %          @�ff���@p���|(��2  Bܙ����@��33���B�z�                                    Bxo�(<  �          @�G�����@^{��
=�K(�B�{����@���9����p�B�.                                    Bxo�6�  "          @�z�\)@:�H��Q��o
=B�aH�\)@�G��fff��B�u�                                    Bxo�E�  �          @�����G�@^{��
=�I��B��H��G�@���9����B�                                    Bxo�T.  �          @�녿Tz�@A����
�d��Bҽq�Tz�@���Z�H�Bȅ                                    Bxo�b�  �          @��ÿ��@+���=q�v�B����@���n�R�!�B�L�                                    Bxo�qz  T          @��þaG�@������B��=�aG�@��
�����1�B�                                      Bxo��   
�          @�z�>�{@�
��  ffB�W
>�{@|(���33�?
=B�.                                    Bxo���  
�          @�\)��@G���  �B�aH��@�z������7�B�                                    Bxo��l  "          @��R=L��@U����H�]�HB�.=L��@�(��QG��33B�                                    Bxo��  W          @�
=?}p�@Z=q��Q��U{B�z�?}p�@�p��J=q� ��B�.                                    Bxo���  	�          @�=q>��H@p  ��\)�M�\B�.>��H@�\)�@����{B�\)                                    Bxo��^  �          @���>�(�@e������T(�B�8R>�(�@���G�����B�L�                                    Bxo��  
�          @���=u@�G���ff�?=qB�aH=u@����(Q����
B���                                    Bxo��  
�          @�G��u@�Q������4��B��R�u@�G������=qB�Q�                                    Bxo��P  �          @�=q=#�
@��������A
=B���=#�
@��,(�����B��                                    Bxo��  T          @���G�@dz���G��Z��B�aH��G�@�{�U�
=B�Q�                                    Bxo��  
�          @��H>#�
@�������2��B��)>#�
@��H���B���                                    Bxo�!B  
W          @�=q��@}p����H�D��B�  ��@�z��1G���\)B��q                                    Bxo�/�  �          @�\)��R@�����G��+��B�W
��R@�Q��Q���{B���                                    Bxo�>�  T          @�
=��G�@���W
=�	��B��)��G�@��Ϳ�=q�NffB�33                                    Bxo�M4  
�          @�{��33@�ff�C33���B�\��33@���u�z�Bʔ{                                    Bxo�[�  
+          @�ff�Ǯ@���<(���B�LͿǮ@���^�R�=qB��H                                    Bxo�j�  
�          @�z����@�\)�7���(�B��)����@�
=�^�R��B�W
                                    Bxo�y&  �          @�����@�ff�j=q�G�B��)����@��׿������Bܞ�                                    Bxo���  �          @�\)���H@w����
�.�B��H���H@�z����z�B���                                    Bxo��r  �          @�Q����@u���\)�3��B��H����@���p���p�B���                                    Bxo��  �          @\�
=@e��G��3�B�Ǯ�
=@�ff�&ff����B�
=                                    Bxo���  �          @�33�
=@l(�����7{B�=�
=@�=q�(Q���Q�B�(�                                    Bxo��d  �          @�(��
=@j�H���9Q�B��)�
=@��\�,(��У�B��                                    Bxo��
  �          @�Q����@hQ������5�
B�  ���@���#�
��(�B�aH                                    Bxo�߰  �          @����p�@r�\�����=�B����p�@���.�R��z�Bؙ�                                    Bxo��V  �          @ȣ׿\@g����
�LQ�Bᙚ�\@��R�Fff���Bԏ\                                    Bxo���  �          @�{��  @������H�?��B��H��  @�\)�+����BȨ�                                    Bxo��  �          @�zῸQ�@�����33�5(�B�.��Q�@�p�������RB���                                    Bxo�H  
�          @�{��
=@�33���H�1z�B��\��
=@��R��H��p�B�8R                                    Bxo�(�  �          @����33@�������1�B�ff��33@�=q�z����B�\)                                    Bxo�7�  �          @��׿E�@}p����R�?�B���E�@�(��%���
=B�G�                                    Bxo�F:  �          @��H�#�
@s�
���
�C�B�aH�#�
@��R�"�\��Q�B�G�                                    Bxo�T�  �          @��?#�
@��H�J=q��
B���?#�
@����
�#�
B�.                                    Bxo�c�  �          @�{>��@�\)�Y���z�B�ff>��@�ff�����c�B��\                                    Bxo�r,  �          @�{?E�@����:=q��Q�B��R?E�@��ÿQ���B�33                                    Bxo���  �          @��H>\)@\)�s33�.p�B��H>\)@����Q����HB���                                    Bxo��x  �          @�Q�J=q@S33���R�cz�B��J=q@��\�`  ��BŔ{                                    Bxo��  �          @�33��@L�����R�m  B�p���@��H�p���ffB��                                    Bxo���  �          @�(��!G�@R�\����i(�Bɨ��!G�@����l(���B��                                    Bxo��j  �          @��Ϳ�@J�H�����o  B��f��@��H�tz��{B��)                                    Bxo��  �          @ʏ\    @Fff��
=�q��B�.    @�  �s�
��B��                                    Bxo�ض  �          @��H>��@#33��ff{B�.>��@��H��\)�)��B�\)                                    Bxo��\  �          @��;�ff@3�
��{�}��B�=q��ff@��\���
�!G�B���                                    Bxo��  �          @�ff���@#33����u�B��)���@������,ffB�z�                                    Bxo��  �          @�\)���@E�����t�\B�aH���@��\�~{��B���                                    Bxo�N  �          @��H�W
=@E���\)�o  BҀ �W
=@�Q��s33�\)B�\                                    Bxo�!�  �          @��H���
@E���z��fB��
���
@��
�^�R���B���                                    Bxo�0�  �          @\��{@J�H����a�Bڀ ��{@��XQ��
=B�W
                                    Bxo�?@  T          @�Q�J=q@I�����\�jz�B�.�J=q@����hQ��{B��                                    Bxo�M�  �          @��0��@%���33�HBѸR�0��@��R���\�)(�BĔ{                                    Bxo�\�  �          @�\)�h��@3�
���H�op�B��H�h��@���a���RB�Q�                                    Bxo�k2  �          @�녿xQ�@QG���z��PB���xQ�@�  �-p��陚B�{                                    Bxo�y�  �          @\�Q�@g�����O�
B͙��Q�@�
=�8����G�B��f                                    Bxo��~  T          @�(�=�\)@S�
��(��d��B��{=�\)@��H�W���
B�k�                                    Bxo��$  S          @��?˅@33���\�}�B]Q�?˅@�33���(�B��H                                    Bxo���  �          @У�@w
=��\���\�R(�C�5�@w
=?�=q��33�EQ�A�ff                                    Bxo��p  �          @�@�G���\)����Az�C��=@�G�?}p������B�APQ�                                    Bxo��  �          @�ff@��\�.{�����D33C���@��\?�
=��z��<
=A�{                                    Bxo�Ѽ  �          @�Q�@��
�8Q�����H=qC�H@��
?�{����@�A��\                                    Bxo��b  �          @�  @������  �IQ�C��@��?\�����>�A��
                                    Bxo��  �          @�p�@�G��(����
=�=C��f@�G�?������\�6ffA�\)                                    Bxo���  �          @�z�@|�Ϳ#�
��\)�L�HC�aH@|��?�Q������CQ�A��R                                    Bxo�T  �          @���@����\)��ff�5��C��@��?����
�>��@�Q�                                    Bxo��  �          @�{���@��H��R��G�B�����@�  �
=��33B�\)                                    Bxo�)�  �          @����P��@�Q쿰���_�B����P��@�?
=@��
B���                                    Bxo�8F  �          @��H�p��@����(��B=qC�=�p��@��
?!G�@�\)C�3                                    Bxo�F�  
�          @�G��g�@��H���
�{
=C���g�@�33>�\)@6ffC�R                                    Bxo�U�  �          @�ff�Tz�@|(�����C)�Tz�@��H�
=��B��q                                    Bxo�d8  �          @��R�XQ�@xQ���H�ɅC\�XQ�@��ÿ���ÅB�k�                                    Bxo�r�  �          @�Q��G�@vff�:�H��(�C  �G�@��R�����+�B���                                    Bxo���  �          @�Q��8Q�@s�
�N{��RB�=q�8Q�@��������X��B�                                      Bxo��*  �          @���'
=@u�Vff�33B�aH�'
=@�(����H�k�
B�=q                                    Bxo���  �          @������@g��z=q�+�RB�G����@�p���
��  B��                                    Bxo��v  �          @�����@_\)��Q��==qB�Ǯ��@�{�����
B�.                                    Bxo��  �          @�녿�\)@Q���  �KffB�3��\)@�33�.{��{B�                                      Bxo���  �          @��\����@G���z��S�HB�33����@����:�H���B�                                      Bxo��h  �          @����\)@C�
��
=�W{B�R��\)@�  �@�����Bٔ{                                    Bxo��  �          @�=q�\)@.�R��33�R  Ch��\)@����B�\����B���                                    Bxo���  �          @����Q�@
�H���
�\z�C	���Q�@z=q�R�\���B�R                                    Bxo�Z  �          @�=q����?޸R��(�(�B�\����@u������8�B���                                    Bxo�   �          @�녿Q�@�R�����{B�8R�Q�@���w
=�'  B�Ǯ                                    Bxo�"�  �          @��׿�G�@�����H�x{B�uÿ�G�@���g
=�(�BԊ=                                    Bxo�1L  �          @��׿O\)@{�����|��B�=q�O\)@���hQ���B�Ǯ                                    Bxo�?�  �          @�����
@)�������}ffB��R���
@�(��j=q��\B�k�                                    Bxo�N�  �          @��\)@#33���\ǮB�B��\)@���p  ���B��                                     Bxo�]>  �          @�p�>��@���z�ffB��q>��@��xQ��$�RB��                                    Bxo�k�  �          @�=q>L��@z����H��B�  >L��@��R���\�)z�B���                                    Bxo�z�  �          @�ff?�z�?!G����
A���?�z�@E��33�az�B���                                    Bxo��0  �          @ƸR@Q�=�Q���=q��@�
@Q�@#33��\)�fQ�BH�R                                    Bxo���  �          @�?�ff>�G���z��A[�
?�ff@333����_�Be33                                    Bxo��|  �          @�{?��?���  W
A��?��@>{��ff�cQ�B��                                    Bxo��"  T          @�  ?aG�?
=��z��{B��?aG�@C33��=q�g��B��                                    Bxo���  �          @��?&ff?����(�¢��B+  ?&ff@C�
�����i  B��                                    Bxo��n  �          @��׾���?�����\)�B�uþ���@z=q����C��B��q                                    Bxo��  �          @�ff=���?�����\)Q�B��=���@w���  �E�B�#�                                    Bxo��  �          @��>��
?B�\����£�RB�33>��
@J�H��z��c�B�k�                                    Bxo��`  �          @�{�S�
@P���G
=�33Cz��S�
@��ÿ�z��i�B��                                     Bxo�  �          @�z���(�@TzῸQ��c\)C\)��(�@g�>�?�  C                                      Bxo��  �          @Å�?\)@k��i���Q�C.�?\)@�z���H���\B���                                    Bxo�*R  �          @ə���
=@%�����|z�B�#׿�
=@�  �|(��=qB��                                    Bxo�8�  �          @�  ?Y��@0  ��z��w�\B���?Y��@���j�H���B���                                    Bxo�G�  �          @�z�=�\)?�33������B���=�\)@��R��z��8��B�                                    Bxo�VD  �          @����?W
=����¡�B�p���@W�����`p�BÊ=                                    Bxo�d�  T          @�
=�(��?h����33�B��׿(��@e����\�^�\B���                                    Bxo�s�  T          @�  �#�
?xQ�����¢�B��#�
@j=q���H�^=qB�                                    Bxo��6  �          @��ÿ�\?��
���
� B����\@mp���G��Z�
Býq                                    Bxo���  �          @�z�Ǯ@���Q�  B����Ǯ@�
=��z��%�HBؙ�                                    Bxo���  �          @�����H?�
=�Ǯ�
B�  ���H@�
=�����7ffB�G�                                    Bxo��(  �          @����(�@����33u�B�\��(�@�=q��{�&��B�k�                                    Bxo���  �          @�ff���R?�33�ə�z�B�(����R@�
=���\�8��B��                                    Bxo��t  �          @ҏ\���H?��R����{B�zῚ�H@������433B��                                    Bxo��  �          @ҏ\��p�@����R8RB�Ǯ��p�@�
=���\�$��B��)                                    Bxo���  �          @ҏ\���H@E����
�k{B�׿��H@���l(���B��                                    Bxo��f  �          @У׿��@S�
���b�B��ÿ��@��
�Z�H����B���                                    Bxo�  �          @�Q쿽p�@j=q����RQ�B����p�@���AG���G�B��)                                    Bxo��  T          @�Q쿌��@s�
��z��QG�B�33����@�{�:�H��(�B��H                                    Bxo�#X  �          @��ÿ�{@q���(��L\)B��쿎{@�G��,����Q�Bʮ                                    Bxo�1�  �          @ƸR��=q@�������==qBѽq��=q@������
=Bɔ{                                    Bxo�@�  �          @�{��@p������
=B����@���w
=�(�B�{                                   Bxo�OJ  �          @����C33@
=�`���)\)C.�C33@k��ff����C��                                   Bxo�]�  �          @��H���?���@ ��A�C#^����?�\@   A�G�C.}q                                    Bxo�l�  �          @�33��ff?�\)@`  B��C"�
��ff����@p  B��C88R                                    Bxo�{<  �          @�(���=q@��@<(�A�C޸��=q?(�@eB�C,n                                    Bxo���  �          @�����@ff@4z�A��HC�����?�R@]p�BffC,c�                                    Bxo���  �          @�������?޸R@QG�B��C
����=�Q�@mp�B��C2޸                                    Bxo��.  �          @�����?=p�@I��B�C+O\��녿@  @I��Bz�C<��                                    Bxo���  �          @��
��Q�>�p�@Z=qB{C/�=��Q쿙��@Mp�B��CB�                                    Bxo��z  �          @��
���
?}p�@r�\B �HC'B����
�Q�@u�B"��C>�)                                    Bxo��   �          @�(�����?�=q@qG�B�C"�)������\@~{B)��C:��                                    Bxo���  �          @�p���ff?�@[�B��C����ff=���@x��B$��C2��                                    Bxo��l  T          @�(���{?�Q�@�HA�(�C����{?#�
@B�\A���C,�
                                    Bxo��  �          @��
����@z�@+�A�(�CxR����?\(�@\(�B{C)��                                    Bxo��  �          @����{@33@'
=A�z�C�{��{?^�R@W�B  C)p�                                    Bxo�^  �          @�z�����?��@C33A�
=C!�����=��
@]p�BC3                                      Bxo�+  �          @������?��
@Q�BffC$�������{@`  BQ�C8(�                                    Bxo�9�  �          @����J�H@^{@\(�B�RCh��J�H?��
@�z�BS33C5�                                    Bxo�HP  �          @�G���z�@z�H@y��B*
=B����z�?�(�@�  B�(�C�3                                    Bxo�V�  �          @���@��@VffBffB�k��@
=@�
=B`�C                                    Bxo�e�  �          @����ff@�
=@P��B\)B�  ��ff@\)@�{BhQ�B��)                                    Bxo�tB  �          @��R����>L��@g
=Bp�C1p��������H@S33B
z�CE�{                                    Bxo���  �          @\����?B�\@��
B-�\C*�������H@�Q�B'�
CC�q                                    Bxo���  �          @�33���?���@xQ�BQ�C�\��녾�@�G�B3p�C5�q                                    Bxo��4  �          @�33��?��@��HB*33C� ���   @��HB7�C:�
                                    Bxo���  
�          @��H��
=?���@\(�Bp�C���
==�\)@y��B$\)C3�                                    Bxo���  �          @Å��  ?�ff@�z�B8�C"���  �\(�@�  B>��C@(�                                    Bxo��&  �          @Å���R?�  @c�
B��C"G����R����@vffB�HC7�H                                    Bxo���  �          @��
��(�?�@.�RA�G�C ���(�>�33@P��Bz�C0\                                    Bxo��r  �          @�(����?���@2�\A�Q�C%�������Q�@FffA��HC4��                                    Bxo��  �          @��
���?�(�@;�A��HCp����>���@`  B  C/k�                                    Bxo��  �          @�����@�
?�33A}C33���?��@=qA��RC(�H                                    Bxo�d  �          @�Q�����?�G�@�A�ffC!�\����?   @5�A�33C.��                                    Bxo�$
  �          @�=q��p�?���?��A�z�C 33��p�?\(�@$z�Aȣ�C*�q                                    Bxo�2�  �          @�33��\)?���@�A�=qC����\)?��@C�
A�33C-s3                                    Bxo�AV  �          @Å����?�@#33A��HC"������>�z�@A�A홚C0��                                    Bxo�O�  �          @�(���  ?���@
�HA�G�C%@ ��  >�=q@&ffA�  C1(�                                    Bxo�^�  �          @�z����\?�z�@�A�(�C#k����\?�@$z�A�\)C.��                                    Bxo�mH  T          @����@<��?xQ�AC�3���@	��@�RA�z�C�f                                    Bxo�{�  �          @�p���Q�@�R?��A*�HCǮ��Q�?�z�@
=qA��HC#:�                                    Bxo���  �          @�(�����@{?�Q�AZ{C�����?��@�
A���C&�{                                    Bxo��:  �          @�p����H@>{>\)?��C�R���H@#�
?�G�Adz�Ck�                                    Bxo���  �          @�(����@Q녿���z�CQ����@H��?���A"�\Cn                                    Bxo���  �          @�(���(�@4z�>k�@Q�CQ���(�@��?\Ah  C�                                    Bxo��,  �          @\��z�@@  �\�hQ�C� ��z�@4z�?���A%�C5�                                    Bxo���  T          @�G���ff@,��?�ffA5�CǮ��ff?��@(�A�
=C                                      Bxo��x  �          @��H�u�@l���ff��(�C��u�@��
�����QG�CG�                                    Bxo��  �          @��
��(�@QG�� �����CE��(�@u��u�z�C
��                                    Bxo���  �          @�(���{@5���
��z�C���{@]p������RC�
                                    Bxo�j  �          @�33���
@H���Q���
=CJ=���
@qG������|��C+�                                    Bxo�  T          @�Q���p�@Tz����x��C\��p�@j=q>W
=@Q�Cff                                    Bxo�+�  �          @�����ff@(�ÿ����S\)C����ff@<��=�Q�?\(�C=q                                    Bxo�:\  �          @�����
=@B�\���H�B=qC0���
=@P  >\@p��Cn                                    Bxo�I  �          @�G�����@?\)�(���{C�f����@:�H?aG�A33C}q                                    Bxo�W�  �          @�����z�@333���R�K�C#���z�@'�?��A,z�C�{                                    Bxo�fN  �          @����Q�@Mp������R�\C�R��Q�@?\)?��RAF�\C��                                    Bxo�t�  �          @�(����@[�?��
A"{C\)���@!�@!�A��
CB�                                    Bxo���  �          @�z����
@hQ�>k�@��C�����
@E�?�
=A���C@                                     Bxo��@  �          @�p����@QG������z�HC���@Dz�?��HA=�C��                                    Bxo���  �          @�ff��33@_\)��Q��_�C����33@p  >���@xQ�CǮ                                    Bxo���  �          @��H��  @>�R��z��3�
C@ ��  @0  ?�A9p�C0�                                    Bxo��2  T          @�����
=@]p������+\)C8R��
=@c33?8Q�@��C��                                    Bxo���  �          @�Q���=q@hQ쿥��Tz�C
Q���=q@s33?(�@ƸRC�q                                    Bxo��~  �          @��R��=q?�p�?�p�A�G�C!!H��=q?0��@ffA�\)C,=q                                    Bxo��$  �          @��R����@�
@�A���C+�����?5@>�RBp�C+O\                                    Bxo���  �          @������
?��@QG�Bz�C$
=���
��33@b�\BC8)                                    Bxo�p  �          @������?�=q@U�Bz�C�\���    @s33BC3�3                                    Bxo�  �          @�Q���z�@ ��@O\)B��C�\��z�>L��@s�
Bp�C1}q                                    Bxo�$�  �          @����@�@0  A�
=C�����?z�@]p�B�HC,�                                    Bxo�3b  �          @�z����@B�\?+�@ҏ\C�3���@�@33A��C�                                    Bxo�B  �          @�  ��=q@I��?
=q@�C33��=q@\)?��RA�G�C�{                                    Bxo�P�  �          @������
@P��>�@�Q�CB����
@(Q�?�p�A�G�C�                                    Bxo�_T  �          @�\)����@W
=>�(�@���C������@.�R@   A��C�)                                    Bxo�m�  �          @����{@l�ͽ�\)�333C����{@Q�?�p�A��\C{                                    Bxo�|�  T          @�
=��G�@fff?^�RAQ�C�)��G�@.�R@   A�Q�C                                    Bxo��F  �          @�p���p�@333?�  A�33CT{��p�?���@8��A�\)C!��                                    Bxo���  �          @�{��z�@<��?�G�AlQ�C���z�?���@0  A��
CQ�                                    Bxo���  �          @������H@W�?�\)A+�C#����H@��@'�AΣ�C��                                    Bxo��8  �          @�\)����@_\)?c�
Az�C�{����@'�@{A�p�CE                                    Bxo���  �          @�\)��Q�@^{?��
A
=C����Q�@!G�@%AͅC
=                                    Bxo�Ԅ  �          @������@o\)?��AN�\C8R���@'
=@?\)AC�{                                    Bxo��*  �          @����{@l��?�\)Ay�C.��{@(�@N{B�C33                                    Bxo���  �          @�z��w�@qG�@p�A�{C���w�@p�@qG�B!
=C@                                     Bxo� v  �          @�{���@`  ?�\)A���C�{���@��@UB	(�C&f                                    Bxo�  Y          @���(�@���?���AR=qC����(�@5@H��B (�CxR                                    Bxo��  �          @�  ��G�@~�R?�A���Ch���G�@!�@g�B  C�                                    Bxo�,h  T          @�{��(�@s33?�z�A���C	\)��(�@�@a�BG�C)                                    Bxo�;  
�          @�p���33@aG�?�z�A��HC  ��33@Q�@Y��B�C�                                    Bxo�I�  �          @��\��  @U�?�{A�z�C� ��  @
=@B�\A��C��                                    Bxo�XZ  �          @����33@e?�=qAz{C� ��33@ff@H��B��C��                                    Bxo�g   �          @��
��(�@s�
?ǮAy�C	J=��(�@"�\@O\)B�CW
                                    Bxo�u�  �          @�����  @Vff?��
AN{CQ���  @�\@1G�A�  C
                                    Bxo��L  �          @�������?�33?�  AG�C =q����?��@G�A�Q�C(��                                    Bxo���  �          @��R��(�@�?uA"�\C�f��(�?�
=?�z�A���C#�R                                    Bxo���  �          @�33�h��@�p�?�\)A1��C� �h��@R�\@J=qB33C	�3                                    Bxo��>  �          @�  �n{@q�@��A���C���n{@
=q@u�B&�RC��                                    Bxo���  �          @�=q�u�@{�?�G�A��\CL��u�@"�\@^�RBp�Cz�                                    Bxo�͊  
�          @�����Q�@fff@Q�A��HC
)��Q�@z�@h��Bz�C�q                                    Bxo��0  
�          @�33�|��@h��@{A�p�C	^��|��@�
@n�RBCz�                                    Bxo���  �          @��
�s33@~{@�
A�=qC�s33@�H@p��B=qCz�                                    Bxo��|  T          @���c�
@���@
�HA�33C�)�c�
@!G�@|��B(�C��                                    Bxo�"  �          @����n{@���?�  A��HC.�n{@<��@l��B�C��                                    Bxo��  �          @���qG�@�  ?\)@�p�C�3�qG�@hQ�@.{A�{C�                                    Bxo�%n  �          @�\)�>�R@E@q�B%��C��>�R?J=q@�=qBe��C%
                                    Bxo�4  �          @��R�>�R@G�@o\)B#�
C���>�R?W
=@���Bd�HC$L�                                    Bxo�B�  �          @�G��;�?���@�G�BR�HC���;��c�
@�{B]�CD��                                    Bxo�Q`  �          @�(��?\)@J�H@z=qB'�CW
�?\)?J=q@�
=Bhp�C%33                                    Bxo�`  �          @���7
=@L(�@~{B+Q�C�f�7
=?G�@���Bn  C$�
                                    Bxo�n�  �          @��H�5�@7
=@�
=B9G�C�R�5�>�p�@��\Br��C,��                                    Bxo�}R  �          @�\)�%�@7�@��RB=�HC���%�>\@��\B{G�C+��                                    Bxo���  �          @��XQ�@W
=@7
=A��
C��XQ�?�p�@��BA��CaH                                    Bxo���  �          @����C�
@N�R@j�HB�CaH�C�
?s33@���Ba(�C"�f                                    Bxo��D  �          @�G��Mp�@R�\@\(�B(�C^��Mp�?�\)@��
BV��C ޸                                    Bxo���  �          @�Q��r�\@`  @{A���C	Y��r�\?��
@y��B+��C��                                    Bxo�Ɛ  �          @�=q�~�R@s33?�G�A�{CO\�~�R@��@[�B�C�                                    Bxo��6  �          @��H�\)@p��@   A�(�C��\)@\)@g�B=qC�q                                    Bxo���  �          @�(����@Vff@�HAÅC(����?�@q�B �C!H                                    Bxo��  �          @�{���?Q�@l(�B�C)n�����33@fffB33CB�)                                    Bxo�(  �          @�{��?!G�@��\B/�C+k�����G�@uB#G�CG�                                    Bxo��  �          @�z�����?���@X��B��C�=���ͼ�@z=qB-Q�C4c�                                    Bxo�t  �          @�\)����?��@c�
B�C%������Q�@i��B  C>&f                                    Bxo�-  �          @�����>�
=@�G�B*��C.�\�����
=@l(�B33CI�                                    Bxo�;�  �          @��o\)@\)@w
=B"�RCW
�o\)>B�\@��HBK33C1�                                    Bxo�Jf  �          @��
���@��@dz�B  C�)���>\)@�
=B8p�C1�R                                    Bxo�Y  �          @��
��  @�@(�A��HCaH��  ?:�H@P��B��C+Q�                                    Bxo�g�  �          @����  @{@,(�A�p�C@ ��  ?B�\@dz�B=qC*h�                                    Bxo�vX  
�          @��H��Q�?�=q@:�HA�\)C���Q�=���@\(�B	��C2�f                                    Bxo���  �          @\���\?�G�@�A�p�C$33���\=�Q�@3�
A�Q�C2�q                                    Bxo���  �          @�����z�?�  @%�A�p�C!5���z�>B�\@G
=A�p�C1�)                                    Bxo��J  �          @�(����?��H?�=qA��HC {���?B�\@$z�Aə�C,                                    Bxo���  �          @�����
=@��=�Q�?\(�C�{��
=@�
?�=qAL��CO\                                    Bxo���  �          @������
@#33�u�ffC�H���
@+�>�@���C}q                                    Bxo��<  �          @�G����
@'
=�c�
�(�C{���
@,��?\)@��CL�                                    Bxo���  �          @�����=q@Fff�^�R��C�=��=q@G
=?W
=A (�CxR                                    Bxo��  �          @�ff����@4zῢ�\�F�\Cٚ����@Dz�>�33@VffC�3                                    Bxo��.  �          @�{�|��@����=q�w\)C�{�|��@��H?@  @��CJ=                                    Bxo��  �          @������@o\)�ٙ���Q�Cff���@��\>�ff@�G�C�H                                    Bxo�z  �          @�  ��\)@N�R������C=q��\)@{���z��0��C
                                    Bxo�&   T          @�����
@{��H����C����
@W
=�O\)����Cff                                    Bxo�4�  �          @����
=@���8����
=Cp���
=@Z�H�����L(�C
                                    Bxo�Cl  �          @��R���@���U���C�f���@tz��{���HC�                                    Bxo�R  �          @�33�Z�H@|�Ϳ޸R��p�C���Z�H@���?�@��C �H                                    Bxo�`�  �          @�\)�QG�@S33�n{���C�R�QG�@�  ���R�hQ�B�(�                                    Bxo�o^  �          @���A�@Dz��y���(z�C���A�@�(���\���B�W
                                    Bxo�~  �          @���   @qG��_\)�\)B���   @��ÿ�ff�(��B��f                                    Bxo���  �          @�33�ff@^�R��ff�7�
B�L��ff@��Ϳ���ffB�p�                                    Bxo��P  �          @������@0����p��d=qB�B�����@����333���
B�.                                    Bxo���  �          @������@p  ��z��1��B��f����@�33��{�|��B�                                    Bxo���  �          @\����@qG�����F�\BӨ�����@��� �����RBɔ{                                    Bxo��B  T          @��H���@dz���  �1B��H���@�  ������B�                                      Bxo���  T          @�
=�2�\@Dz����
�@=qC@ �2�\@�������G�B�(�                                    Bxo��  T          @��
�5�@   �����]�HCǮ�5�@���Q���B��=                                    Bxo��4  �          @Å�,��?�(����i\)C}q�,��@����c33�z�B���                                    Bxo��  �          @Å� ��?�����H�t�\C��� ��@��\�q���B�=q                                    Bxo��  �          @�(��\)@
=�����f��C� �\)@�
=�S�
��B�8R                                    Bxo�&  �          @�����@=p������K=qC ���@�ff�(���(�B�=q                                    Bxo�-�  �          @�Q��
=@\)�����d��C���
=@����H�����B�\                                    Bxo�<r  �          @�=q��p�?�ff���R�C�ÿ�p�@���w
=��HB���                                    Bxo�K  T          @����\?Q����{CLͿ��\@h������Cz�Bڏ\                                    Bxo�Y�  �          @�ff��=q@*=q���
�op�B���=q@��?\)����B��                                    Bxo�hd  �          @�zῬ��@9����p��c��B��
����@�G��,(���z�B���                                    Bxo�w
  �          @�녿�G�@�R����=qB�׿�G�@�z��U�z�B�u�                                    Bxo���  T          @��\��?�������33C.��@����r�\�#p�B���                                    Bxo��V  T          @�녿�  ?�  ��\)B�LͿ�  @�33�vff�)�B�G�                                    Bxo���  T          @�\)�L��?�
=���R� B�{�L��@�=q�z=q�-Q�B�\)                                    Bxo���  �          @���G�?�������B�����G�@w
=���
�;p�B�(�                                    Bxo��H  �          @�\)�u?�33���\��B��׽u@x������;�B���                                    Bxo���  �          @�\)��(�?��R��z���Cp���(�@��\�tz��'p�B�aH                                    Bxo�ݔ  �          @���aG�?�{�����=B�p��aG�@�ff�qG��'�B���                                    Bxo��:  �          @�(���=q@0  ���
�q�RB����=q@�z��.{��  B�B�                                    Bxo���  �          @�녿}p�@8������fffB�녿}p�@�\)�%��Q�Bʊ=                                    Bxo�	�  �          @�p����@P�������Yz�B��H���@�G��Q���Q�B�=q                                    Bxo�,  �          @��þaG�@
=���
�RB��aG�@���l(��p�B�\)                                    Bxo�&�  �          @��
=L��?�\)��{{B�\=L��@������H�5��B��q                                    Bxo�5x  �          @�p���\)?޸R����B�zᾏ\)@�=q����'��B�{                                    Bxo�D  
�          @��׿J=q?������
�)B�녿J=q@���p���{B�ff                                    Bxo�R�  �          @�p���@{��ff�=BΔ{��@�  �^{�  B���                                    Bxo�aj  
�          @��ÿ:�H@   ��z��|�B�ff�:�H@��\�C�
� ffB��                                    Bxo�p  
�          @�������@?\)����g33B��)����@�\)�1���G�B��f                                    Bxo�~�  (          @�{����@I�����\�\�HB��)����@�33�*�H��z�B�G�                                    Bxo��\  
Z          @��׿�z�@7
=����hffB� ��z�@��
�6ff��B�Ǯ                                    Bxo��  
�          @�{��\@>{���H�Y�HB�q��\@�=q�#�
�˙�Bڏ\                                    Bxo���  
�          @�
=���@�����8RB�8R���@����X���
��B̞�                                    Bxo��N  T          @�G��O\)@�
���H��B��
�O\)@���\�����B�{                                    Bxo���  �          @�33��  @Dz��G�����B�𤿀  @b�\=�?�33B�u�                                    Bxo�֚  "          @�����33@���?�\@�(�BѮ��33@��R@Mp�BBظR                                    Bxo��@  �          @�  ��=q@����ff���RBʸR��=q@�G�@(�AӮB�W
                                    Bxo���            @�p��@  @��ÿ
=q���B���@  @���@�AͮB���                                    Bxo��  �          @�z´p�@�{<�>�p�B�k���p�@�@(��A��B�{                                    Bxo�2  �          @��Ϳ�33@U�@  �!z�B�𤿓33@�p��G��(�B��                                    Bxo��  
�          @��Ϳc�
?�ff��(�=qB���c�
@���p  �%�\B��                                    Bxo�.~  T          @��
�W
=@���G��=B܊=�W
=@�33�Dz��G�BȽq                                    Bxo�=$  
�          @�����ff?��H��p�ffBљ���ff@r�\�6ff��B���                                    Bxo�K�  T          @O\)�\?�(��	���;��Bɣ׾\@3�
�G��up�B�G�                                    Bxo�Zp  �          @c33�
=q@�\�&ff�?ffBνq�
=q@Tz�}p����Bƞ�                                    Bxo�i  T          @tz῰��@��@ffB �B��H����?L��@P��B�RC�                                    Bxo�w�  �          @�����\)@��@X��BN(�B�8R��\)>�\)@�B�C%��                                    Bxo��b  �          @�p����@W�@N�RB&33B��쿱�?��@���B���C�                                    Bxo��  �          @���[�@�Q�=#�
>�C���[�@XQ�@
=qA�Cp�                                    Bxo���  �          @����\(�@l(�?�\)As
=C���\(�@��@G�B�C�                                    Bxo��T  �          @�p��e�@E�@�
A�=qCG��e�?��@eB,Q�C��                                    Bxo���  �          @��R���
@<(�@	��A�33C{���
?���@XQ�BC"�\                                    Bxo�Ϡ  �          @�����33?Ǯ@8Q�A��C!G���33�W
=@QG�B\)C6�{                                    Bxo��F  T          @�������?�ff@,(�A�z�C$� ���׾���@>{A���C7�                                    Bxo���  
�          @�33��z�?:�H@&ffA�{C+�\��z�E�@%A���C<�                                    Bxo���  
�          @������\?p��@<��B{C(n���\�G�@@  B\)C=�)                                    Bxo�
8  
�          @���33?�  @@��Bp�C$���33�
=q@Mp�B�
C;#�                                    Bxo��  �          @�{��Q�?�  @A�BQ�C ����Q쾮{@W
=B��C8�H                                    Bxo�'�  
�          @������?aG�@k�B(�C)����ÿ�(�@e�BG�CC�                                    Bxo�6*  "          @������?���@aG�B=qC�
���׾��R@}p�B+
=C8)                                    Bxo�D�  �          @�{��  @ ��@J=qB��C�f��  <��
@p  B%��C3��                                    Bxo�Sv  "          @�  �~{?@  @p  B+\)C)J=�~{��{@dz�B!�CF�                                    Bxo�b  T          @�=q�z�H?p��@�{B8��C&���z�H��Q�@���B0p�CH8R                                    Bxo�p�  �          @��R�P��?^�R@���BU  C%�P�׿�@���BE��CO(�                                    Bxo�h  T          @ə��E�@��
��33�(Q�B�q�E�@���?��A��B�Ǯ                                    Bxo��  �          @Ǯ�A�@��H�E���{B�\�A�@�@��A�Q�B�{                                    Bxo���  �          @ȣ��=p�@�ff�+����B���=p�@��R@A�z�B�3                                    Bxo��Z  "          @�33�X��@�33�.{��  B��q�X��@��@.{A�33B���                                    Bxo��   T          @����3�
@�����z��L  B�3�3�
@�{?�\A�G�BꙚ                                    Bxo�Ȧ  �          @��
�y��@�
==���?^�RC ��y��@���@/\)A�\)C:�                                    Bxo��L  �          @�=q�k�@��H�z���\)B��k�@��@\)A�  C ��                                    Bxo���  T          @��H�z�H@��
��  �G�C ���z�H@���?�G�A�33C!H                                    Bxo���  �          @�z��hQ�@�������-G�B��)�hQ�@�ff?�(�Az{B�z�                                    Bxo�>  T          @�z��\(�@�������-B��\�\(�@�=q?��A�(�B�Q�                                    Bxo��  �          @�Q��-p�@�\)���-�B螸�-p�@�Q�?�Q�A�p�B�=                                    Bxo� �  �          @�p��33@�
=?��
A��B����33@��@�=qB
=B�.                                    Bxo�/0  �          @Ϯ�5@�G�?z�@��
B�8R�5@��\@eB�B�\                                    Bxo�=�  �          @����333@�\)�u�   B�
=�333@��@AG�A�ffB�Ǯ                                    Bxo�L|  �          @ȣ��.{@�z����eG�B��.{@�(�?˅Al(�B鞸                                    Bxo�["  �          @ƸR�z�@��
�����k�B܊=�z�@��\?�
=A|z�B�                                    Bxo�i�  N          @Ǯ��@�\)�8Q���z�B���@�
=@p�A��
B��H                                    Bxo�xn  "          @�Q��
�H@�Q쿏\)�%�B�=q�
�H@��R@	��A�ffB�G�                                    Bxo��  �          @�{��R@�z�W
=����B�(���R@��R@z�A��
B�W
                                    Bxo���  �          @Ǯ��@��H����?
=B���@��?�z�A���B�k�                                    Bxo��`  �          @����  @�
=��p��4��B���  @�\)@�A�p�B���                                    Bxo��  �          @Ǯ��33@�=q�����V�RBӸR��33@�{?�z�A���B�p�                                    Bxo���  �          @�{��=q@�(�@�A���B͏\��=q@tz�@�{BJ�RB�u�                                    Bxo��R  �          @�
=�8��@'����H��p�C	�)�8��@HQ�#�
���HCǮ                                    Bxo���  "          @�33�L(�?!G���(��fz�C(�\�L(�@R�\�\)�$�C!H                                    Bxo��  �          @\�>�R>�������p�
C+
=�>�R@N�R��{�.��C�3                                    Bxo��D  �          @��
�*�H?\)���}�C(8R�*�H@X�������333C .                                    Bxo�
�  �          @���%�?s33��z��w�C���%�@dz��tz��#�B��                                    Bxo��  �          @�G��9��?}p������h  C!#��9��@^{�fff�{C�)                                    Bxo�(6  �          @��
�#33?aG���ff�z�C ��#33@b�\�z=q�'Q�B���                                    Bxo�6�  �          @����33?�R��G�p�C$�{�33@`  ���\�7B��=                                    Bxo�E�  �          @�
=�=�G���Q��3C1W
�@L(���G��I�
B��\                                    Bxo�T(  �          @���'
=�����
=�fC7G��'
=@4z���{�J�RC�\                                    Bxo�b�  �          @�{�5@G�����]��C���5@��J�H���RB�ff                                    Bxo�qt  "          @��H�7
=?�33��Q��^33CxR�7
=@����L����(�B�aH                                    Bxo��  
Z          @�=q�(�@������bG�C���(�@�z��A���=qB�ff                                    Bxo���  �          @�
=�P��@!G������833CL��P��@��  ����B�                                    Bxo��f  T          @���N{@+�����333CG��N{@�  ��
����B�=q                                    Bxo��  �          @��5�@I�����
�0�
C��5�@��
��\����B�B�                                    Bxo���  T          @�����H@J=q��G��C
=B��)��H@�������\B��                                    Bxo��X  �          @�=q��@8Q���(��W{B��=��@���#�
��  B�ff                                    Bxo���  �          @�=q��  @g���G��A��B���  @�  �����{B�\)                                    Bxo��  �          @��
��p�@���e�p�B�uÿ�p�@�  �=p��陚B��                                    Bxo��J  �          @�\)��@��\�W��  B�B���@��R�   ��33B��                                    Bxo��  T          @���(�@X���}p��1�RB��R�(�@����  �v�HB�ff                                    Bxo��  �          @��
�z�?����z��uz�C	�)�z�@��
�S�
�	�\B�Ǯ                                    Bxo�!<  �          @�ff��?��H��ff�s�\C����@�  �\����HB�=                                    Bxo�/�  
�          @��H��?�����33�[z�C8R�H��@n�R�W��
�\C&f                                    Bxo�>�  
(          @���	��@^�R�G�����B�k��	��@���>.{@
=qB�=q                                    Bxo�M.  �          @�
=��33?�����t�
C�f��33@�33�5��B�Ǯ                                    Bxo�[�  �          @�Q�>�\)@,(���
=�kG�B�{>�\)@���
��  B��                                    Bxo�jz  �          @�z�?�\?�G���=q�B��?�\@����xQ��'�B��                                     Bxo�y   �          @��>���?\(����
¢�B���>���@y����\)�Cp�B�G�                                    Bxo���  �          @��
���?\)��=q§�)B��ᾅ�@g���33�N�HB��                                    Bxo��l  �          @��;�>�(��Å¨.C	���@j=q����TG�B�k�                                    Bxo��  �          @�ff��p�>�\)��p�«W
C�쾽p�@dz���G��Z33B��                                    Bxo���  �          @�z�<��
?!G����\§�{B��q<��
@k�����Lp�B�B�                                    Bxo��^  �          @��\�8Q�>��H����©�B��8Q�@c33���
�Q�RB��                                    Bxo��  �          @Å�\>�G���G�¨��Cff�\@hQ����H�S�
B��)                                    Bxo�ߪ  �          @�=q=L��?�{�����B��==L��@��\��\)�1p�B���                                    Bxo��P  �          @�������@����G��B�\)����@�Q��W���\B��f                                    Bxo���  �          @��H��p�@.�R�����{�
B�uþ�p�@�
=�Dz���G�B�#�                                    Bxo��  �          @\�B�\@)����ffB�=q�B�\@��J=q����B�8R                                    Bxo�B  �          @�33��=q@3�
��z��y�B�
=��=q@����AG���  B��f                                    Bxo�(�  �          @����p�@-p���=q�{G�B�p���p�@����@����{B�#�                                    Bxo�7�  T          @�
=��\@L(���Q��d�B�\��\@�z��\)�ŅB��q                                    Bxo�F4  �          @���  @:�H��z��q(�B��3��  @���0  ���B�k�                                    Bxo�T�  �          @�ff����@1G���{�v�
B�zᾨ��@�z��7����HB�L�                                    Bxo�c�  �          @���>�\)@,(����
�|��B�=q>�\)@�p��C�
����B��)                                    Bxo�r&  �          @�z�>u@E������n��B�8R>u@�{�1���Q�B�
=                                    Bxo���  �          @ʏ\>��@p����\B�#�>��@��R�c�
�	\)B�                                    Bxo��r  �          @ə�=L��@>{��Q��v��B�=q=L��@�\)�A���B�
=                                    Bxo��  �          @�����33@��\��ff�=\)B�녾�33@��H���
�k\)B��                                    Bxo���  �          @�G��L��@y����Q��=��B��H�L��@��H��p��l��B��                                    Bxo��d  �          @��׾��@�z������7=qB�8R���@�����{�S33B�                                    Bxo��
  �          @�����@��H�y��� p�B����@�
=�L����=qB��H                                    Bxo�ذ  �          @��H��ff@�(��z=q� =qB�\��ff@�Q�J=q��RB��=                                    Bxo��V  �          @�  ���
@��H�u��=qB�
=���
@��@  ����B�8R                                    Bxo���  �          @�ff>�@�z��>{��ffB�.>�@�>�\)@)��B��\                                    Bxo��  �          @��R����@��8Q���B�aH����@�p�@p�A�{B�p�                                    Bxo�H  �          @�zῑ�@���<#�
>�B�Q쿑�@�33@HQ�B ��B�k�                                    Bxo�!�  �          @�G��\@���>��@#�
BҔ{�\@���@N{B��B�                                    Bxo�0�  �          @�Q쿮{@��
�W
=�p�B�LͿ�{@�z�@.{A�\)BԔ{                                    Bxo�?:  �          @Å�
=@`  ��
=�Z{B��
=@�(���\���\B��H                                    Bxo�M�  �          @����\)@��������B΅�\)@���Vff�B��H                                    Bxo�\�  �          @�p�<�@  ��ff�
B���<�@�\)�e���HB�ff                                    Bxo�k,  �          @�{>�{?����(�ǮB���>�{@����{���\B��
                                    Bxo�y�  �          @ƸR?
=q@G����\��B�{?
=q@�33�tz��G�B�Q�                                    Bxox  �          @�    @(���Q�\)B��)    @��R�j�H��B��                                    Bxo  �          @ƸR>u@���\W
B���>u@���q��G�B�L�                                    Bxo¥�  �          @�>���?�33���G�B�.>���@�Q��y����HB�aH                                    Bxo´j  �          @��H?�  @����B�Bn  ?�  @����a��33B�                                    Bxo��  �          @�
=?5@p����{B�ff?5@��\�Z�H�33B�=q                                    Bxo�Ѷ  �          @��>��R?�=q��33G�B��
>��R@���n{��B�B�                                    Bxo��\  �          @��R?E�?������
Bo��?E�@�\)����.�
B�B�                                    Bxo��  �          @�\)?(�?�ff����RB�.?(�@�����  �'�B�Q�                                    Bxo���  �          @�p�>��?�(���ff�B���>��@����  �+\)B��                                    Bxo�N  �          @�\)>���?�\��{��B��H>���@�=q�u���B��                                    Bxo��  �          @�G�>�{@����{B�aH>�{@��
�c33�(�B�\                                    Bxo�)�  �          @��H=L��@�
��(��B��=L��@�33�Q��ffB���                                    Bxo�8@  �          @���mp�@l(���p����C��mp�@�>���@VffC��                                    Bxo�F�  �          @�{�@  @u�O\)�ffB����@  @�\)�&ff��p�B��                                    Bxo�U�  �          @�z��[�@g
=�#�
���C���[�@�p��W
=�
=qB��{                                    Bxo�d2  M          @���QG�@e�-p����CG��QG�@����{�a�B��                                    Bxo�r�  �          @����
=q@:=q��
=�K�B�G��
=q@���������B���                                    BxoÁ~  T          @��H�xQ�?�z���
==qB���xQ�@�33�mp��(�B�33                                    BxoÐ$  T          @�=q<#�
?���\)©u�B�k�<#�
@a���G��PffB�                                    BxoÞ�  �          @�zῆff@\)���ffB�33��ff@��R�L���=qB��                                    Bxoíp  
�          @����G�?�Q���{�)B����G�@���xQ��"��B��                                    Bxoü  	�          @��׼�?����ffk�B��ͼ�@���r�\�ffB��R                                    Bxo�ʼ  	�          @�G���  @�����\)B�G���  @�
=�XQ����B��)                                    Bxo��b  
�          @�ff=��
@+�����{�B��3=��
@��\�>{���B�B�                                    Bxo��  
�          @��
�
=@5������pp�B˔{�
=@��H�-p���
=B�8R                                    Bxo���  T          @�{����@<(���(��p�B��\����@���/\)��Q�B��                                     Bxo�T  �          @�������@333��  �eG�B�ff����@�p��\)���B�p�                                    Bxo��  
�          @�33��
=@XQ�����?��B�=��
=@�p��������
B���                                    Bxo�"�  �          @��H��@{����Z�C� ��@�(��*�H�ڣ�B�u�                                    Bxo�1F  �          @��H�z�@J=q�����=B��3�z�@��R��33��
=B�8R                                    Bxo�?�  T          @�p��\)@C33�����>�C8R�\)@�(���p���33B�                                      Bxo�N�  T          @�{���@����H�WC�
���@��'��ۙ�B��
                                    Bxo�]8  �          @������@-p���Q��\�B�  ���@��H�#�
�хB�                                    Bxo�k�  T          @����@>{�����L��B����@��p����B���                                    Bxo�z�  �          @�G���@�
�����q�RB��{��@����?\)��B�p�                                    Bxoĉ*  
�          @�G��p�@���p��Y=qC	G��p�@��.{��p�B�.                                    Bxoė�  �          @��\�ٙ�@Mp������M=qB�Ǯ�ٙ�@�z��ff���B؏\                                    BxoĦv  
�          @����{@J�H�����@{B�(��{@�\)��
=����B��                                    Bxoĵ  �          @�
=�@(�������U\)C L��@�������33B�B�                                    Bxo���  �          @��ÿ�=q?�ff���Q�B�  ��=q@��
�aG��\)B��H                                    Bxo��h  
(          @�Q쿬��?����  G�B��H����@�p��Z�H�  B��                                    Bxo��  �          @�p���33@33����fB�{��33@�  �P  ��HBиR                                    Bxo��  "          @�z���\?���.{�n�\C�q��\?�z�>�  @�ffC��                                    Bxo��Z  "          @�33�?\)?�  @��HBF�C���?\)�8Q�@�(�B[�CA}q                                    