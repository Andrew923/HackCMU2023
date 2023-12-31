CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230210000000_e20230210235959_p20230211021654_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-11T02:16:54.375Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-10T00:00:00.000Z   time_coverage_end         2023-02-10T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxc��  T          @\�P  �Dz���G��(�C_W
�P  �O\)��Q��`��CA�                                    Bxc�&  �          @����Mp��7
=�l���#
=C]���Mp��G�����Y��CA��                                    Bxc��  T          @��H�Z=q�:=q��\)�4(�C\xR�Z=q����=q�d��C;��                                    Bxc�r  �          @����@  �:=q�{��-  C`.�@  �8Q����H�f  CA��                                    Bxc�-  T          @׮���
�������/(�CM�3���
>��R����E��C/�                                    Bxc�;�  �          @�����z���R�����'Q�CK33��z�>�p����:p�C/}q                                    Bxc�Jd  T          @�p����R��z������.��CGff���R?G�����9=qC*��                                    Bxc�Y
  
�          @ᙚ���R��  �����5��CE�3���R?��
��z��;Q�C'�                                    Bxc�g�  T          @�33��������G��@�
C@�R���?����p��;{C!xR                                    Bxc�vV  "          @޸R���W
=�����F�\C>����?�  ��=q�;\)Cu�                                    Bxc���  �          @�{��ff�����G��G
=C:���ff@�
��z��3�RC#�                                    Bxc���  "          @�z���
=�������PG�C9aH��
=@������9z�Cz�                                    Bxc��H  �          @��H�xQ�z���33�[C<� �xQ�@
=��
=�F�HC}q                                    Bxc���  �          @�ff�z�H�L����z��W��C6�R�z�H@ff����;33C\                                    Bxc���  T          @��
�`  =L������bQ�C3+��`  @!���p��<��C&f                                    Bxc��:  �          @�p��s33>������R�\=qC/}q�s33@4z���{�2��Cn                                    Bxc���  "          @�p��s33>����
=�\�RC1�R�s33@,(���Q��6�C�                                    Bxc��  �          @��j=q?&ff�����`�
C)�q�j=q@J�H��33�.Q�C)                                    Bxc��,  �          @ٙ��j�H?5��{�bC)��j�H@R�\��{�.p�C
�                                    Bxc��  
�          @׮�aG�?s33����e  C$ٚ�aG�@_\)����*�\C8R                                    Bxc�x  T          @��H�h��?�R��  �e33C*T{�h��@N�R��G��2=qC
\)                                    Bxc�&  T          @��
�e?Tz������fp�C&���e@\������.z�C�                                    Bxc�4�  "          @�{��녾\��=q�E�C9
���@�����1p�C�
                                    Bxc�Cj  �          @ָR��z�Tz�����@�C>����z�?�����G��6�C                                     Bxc�R  �          @�����
��\)��  ��CFs3���
?����Q��*�C-�
                                    Bxc�`�  �          @��������=q���H�-�C@5����?�ff�����+z�C%^�                                    Bxc�o\  �          @�p���Q�>�Q���{�7C/���Q�@'���ff��RC0�                                    Bxc�~  T          @�z���\)?�������;�C"����\)@dz��s33�G�Cz�                                    Bxc���  �          @��
����?�ff��=q�:�HC {����@g
=�c33��C�q                                    Bxc��N  �          @�{��  ?:�H��{�Cz�C*B���  @<��������RCG�                                    Bxc���  T          @�33�N{�L����33�/(�C`�\�N{�O\)���H�h�CB\                                    Bxc���  T          @љ��vff�����H�?G�CPc��vff>����  �V=qC-�q                                    Bxc��@  T          @�\)��=q����33�F(�C;���=q?�33�����5(�C@                                     Bxc���  �          @��H��G��p����\)�G�HC@\)��G�?�����=q�?��C��                                    Bxc��  �          @��H�i���=p����R�:=qC[��i�����
�����fp�C9                                      Bxc��2  �          @���E�`  ��{�:�\Cd�
�E�L����Q��x
=CB��                                    Bxc��  �          @����S33�Dz���ff�D\)C^���S33���������t��C9=q                                    Bxc�~  "          @ڏ\�\���<(���33�AQ�C\z��\�;u��(��n(�C8
=                                    Bxc�$  �          @ۅ�k��9����  �;\)CZ:��k���  �����e�C7��                                    Bxc�-�  T          @����r�\�9�����R�8�\CYn�r�\��=q����b{C8�                                    Bxc�<p  �          @���`  �+�����F
=CYc��`  <��
����kC3��                                    Bxc�K  �          @����c33�6ff�����?p�CZ��c33�L�������i��C7+�                                    Bxc�Y�  �          @ٙ��^{�G
=��{�:�RC]�H�^{�����=q�k�HC;��                                    Bxc�hb  �          @�=q��  �xQ����H�E{C@!H��  ?�{���=Q�C @                                     Bxc�w  �          @����p���p���  �N�
CDk���p�?��H��ff�K�C �                                     Bxc���  �          @�����p���33��G��P
=CCu���p�?����ff�K�C                                    Bxc��T  �          @�����
��33���
�F{CE�=���
?��R����H{C$(�                                    Bxc���  �          @�(������������E33CB����?�
=��33�B\)C"O\                                    Bxc���  �          @�G���zῙ�����
�;��CB����z�?�ff��33�:��C$T{                                    Bxc��F  �          @�������
=��Q��ffCJ@ ���=�Q���  �*z�C3�                                    Bxc���  �          @��H���H������=q� ��CG�3���H>������/p�C.��                                    Bxc�ݒ  T          @�33��G���  ���R�'�CD�
��G�?W
=��(��.��C*��                                    Bxc��8  �          @�
=���Ϳ��\�����CA.����?�  ���!��C)��                                    Bxc���  �          @�G���=q��\)��33�&��C?����=q?��R��=q�%��C&�                                    Bxc�	�  �          @�����(���33�����.��C@����(�?������-G�C%��                                    Bxc�*  �          @陚���Ϳ�����G��$33C?:�����?�  ��  �"ffC&��                                    Bxc�&�  
�          @�G���\)�(�������#=qC:�{��\)?�{�������C#��                                    Bxc�5v  "          @�(�����E������+G�C<&f���?�\)����"��C#+�                                    Bxc�D  ^          @�����p������p��&��C>�{��p�?�=q��33�#��C&+�                                    Bxc�R�  |          @�z���{�1G����H� z�CR�{��{��Q���33�B�RC8^�                                    Bxc�ah  �          @�\��{����{�&z�CO\)��{�#�
�����A��C4
                                    Bxc�p  �          @��H������
��z��$��CM�����=��
���=\)C3{                                    Bxc�~�  �          @�=q���H�;���Q����CT�����H�
=q��33�D�
C:�R                                    Bxc��Z  �          @�=q�����(���\)�'��CL������>W
=��ff�=��C1p�                                    Bxc��   �          @���
=�/\)��z��!�CR(���
=���
��(��BC7�
                                    Bxc���  �          @����33�5����\���CRJ=��33��(����
�?33C9�                                    Bxc��L  �          @�  ���\�g
=�������CX� ���\��\)��
=�8z�CC�H                                    Bxc���  �          @������z=q�i����  C[�����������G��333CH��                                    Bxc�֘  �          @����Q���  �e����
C_T{��Q��
=q���
�6��CM��                                    Bxc��>  �          @��
��\)�����dz���C_����\)�����
�7(�CM��                                    Bxc���  �          @�(����R�q��g
=���C\@ ���R��(���{�7��CI�                                    Bxc��  T          @��
��33��z��S�
����Cb���33�����{�8�CRT{                                    Bxc�0  
�          @��H��  ����\(���C`���  �ff���8��CNE                                    Bxc��  �          @ۅ����p��^{��p�C`����������:�
CO
                                    Bxc�.|  �          @��
��(���ff�6ff���Cd����(��9����p��+\)CW\                                    Bxc�="  �          @�(��z=q���R�+�����Cg�z=q�Mp���z��*  C[\)                                    Bxc�K�  T          @��H�w
=�����1���(�Cg���w
=�G
=��ff�-��CZ޸                                    Bxc�Zn  T          @��
�s�
��Q��-p���=qCh��s�
�O\)���,�C\ff                                    Bxc�i  
�          @�Q��p����z��U����Chp��p���7
=���>�\CY33                                    Bxc�w�  �          @߮�p�������C�
��Ci+��p���Fff��  �6�C[xR                                    Bxc��`  T          @ٙ��S�
����e�� Q�Cj�)�S�
�"�\�����O=qCYu�                                    Bxc��  �          @�33�Dz���
=�Z�H���Cl���Dz��*�H���OffC\�q                                    Bxc���  �          @�z��7
=���R�Vff��\Cp�7
=�:�H��\)�O��Ca��                                    Bxc��R  T          @��
�L����\)�;�����CmQ��L���G����
�=C`W
                                    Bxc���  
�          @���S�
��=q�H����(�Ck�\�S�
�8����\)�BffC]!H                                    Bxc�Ϟ  �          @��
�=p����R�333��=qCpc��=p��X�����H�<ffCd�H                                    Bxc��D  �          @��<����Q��7
=����Cp�<���Z=q��p��>
=Ce0�                                    Bxc���  �          @�\)�<����\)�B�\����Cp���<���S�
��=q�Cz�CdL�                                    Bxc���  T          @����(Q����H�=p��ӅCs���(Q��\(���G��EQ�Ch��                                    Bxc�
6  "          @�ff�/\)��Q��u��RCp!H�/\)�"�\��=q�`�RC^��                                    Bxc��  
�          @ָR�Tz����R�U���HCj���Tz��-p���33�G��C[33                                    Bxc�'�  �          @ָR�p������X�����
Ce@ �p���Q���\)�A  CTY�                                    Bxc�6(  �          @�ff�hQ���33�K����
Cg���hQ��,(������=G�CXxR                                    Bxc�D�  "          @ҏ\�G
=��(��+���Q�Cn���G
=�X����{�633Cc��                                    Bxc�St  �          @Ӆ�A������ ������Cp
�A��e����1��Ce�{                                    Bxc�b  T          @��X�������P����ffCi�H�X���,����  �D(�CZ�=                                    Bxc�p�  �          @�
=�tz��s�
��  �{C`���tz�У������O=qCK
                                    Bxc�f  T          @�(��[���33�E����Cg�[��!G����R�?z�CXL�                                    Bxc��  �          @�z��@��\��G���(�B�{��@�(�>�@�B��H                                    Bxc���  �          @�G��!�@"�\�G��$(�C�R�!�@j=q����(�B�aH                                    Bxc��X  "          @�
=���@������B΅���@��>u@3�
B�k�                                    Bxc���  ^          @���\)@��H��\��Q�B����\)@�������B�p�                                    Bxc�Ȥ  |          @����G�@~{�'
=���B�G�@��R�z���=qB�z�                                    Bxc��J  T          @�G��.{@c�
�3�
� (�B����.{@�{�n{�"�HB�{                                    Bxc���  �          @���O\)@Mp��9�����C33�O\)@�p����H��B��                                    Bxc���  
�          @�G��`  @7
=�L���
=C�R�`  @~�R������=qCQ�                                    Bxc�<  ^          @�
=�aG�@$z��U����C���aG�@q녿�{��=qC��                                    Bxc��  �          @������@33�J�H�\)C�)���@O\)��Q����Ck�                                    Bxc� �  T          @�33���
?Q��A��=qC)�3���
@z��ff��\)C޸                                    Bxc�/.  �          @�(�����=��
�<(���C2�f����?����&ff��C"�f                                    Bxc�=�  T          @��R��ff=�G�����G�C2�
��ff?��
������ffC&��                                    Bxc�Lz  �          @�����Q�u��Q���\)C6���Q�?#�
��������C,aH                                    Bxc�[   �          @�p���p��E��G���Q�C=\)��p�>B�\�	����\)C1�)                                    Bxc�i�  �          @�\)��zῇ���Ə\CA�=��z�����z�C4h�                                    Bxc�xl  �          @�=q���׿�����\���CE���׿��{��ffC:��                                    Bxc��  T          @��
�W�?\�n�R�5(�C�R�W�@@  �,(���=qC
G�                                    Bxc���  �          @��\�a녾��H�e��4z�C;޸�a�?����[��+�C"B�                                    Bxc��^  �          @��Z=q��  �W
=�-�RCDh��Z=q?��]p��4G�C*��                                    Bxc��  T          @��\)�h���s33�+p�C@�)�\)?W
=�s�
�,Q�C(
                                    Bxc���  �          @�G����ÿW
=�����8G�C?�����?�(���{�3�C$\                                    Bxc��P  T          @��H�qG������=q�?�CC��qG�?n{��33�@��C&33                                    Bxc���  �          @�33�S33���H�w
=�8��COY��S33>����
=�O�C1c�                                    Bxc��  �          @�{�e��ff�fff�'�\CN���e�L�������A  C4                                    Bxc��B  �          @�z��n�R���R�e��&��CI�\�n�R>aG��w��8=qC0��                                    Bxc�
�  �          @�z��qG���  �c33�$�\CI���qG�>B�\�vff�6{C1)                                    Bxc��  "          @��Z�H����U��)�CH���Z�H>�\)�c�
�7�C/B�                                    Bxc�(4  �          @�z��^�R��  �Dz��
=CKT{�^�R���
�Z�H�1�C4W
                                    Bxc�6�  �          @�G��r�\��G��a��(
=CB���r�\?�R�g��-=qC*��                                    Bxc�E�  �          @�
=�dz����z����CO�)�dz�333�:=q�ffC?{                                    Bxc�T&  �          @���
=q��z�Ǯ��G�CsǮ�
=q�r�\�  ��{CpL�                                    Bxc�b�  �          @�  �����\�����Cv\���u� ����z�Csff                                    Bxc�qr  T          @�����������W
=�'�CvY������j=q��p�����Cs��                                    Bxc��  �          @�  ��=q���
�\)��
=CwG���=q�x��� ���ȣ�Ct�                                    Bxc���  �          @�  �����G�>8Q�@
�HC}������������
C|+�                                    Bxc��d  T          @�p������\)?@  A�C�p���������  �yp�C�E                                    Bxc��
  �          @�Q���
�}p�>#�
@z�Cr�=��
�i���Ǯ���Cp��                                    Bxc���  �          @�ff�a��mp��Tz��33Cbp��a��@  ��֣�C\T{                                    Bxc��V  �          @����xQ��X�ÿ��H�S�
C]#��xQ��"�\�#33��=qCU:�                                    Bxc���  
�          @����tz��W�����k
=C]u��tz���R�)�����
CT�R                                    Bxc��  �          @�{�r�\�j�H��ff�]p�C`{�r�\�0���0  ��RCX{                                    Bxc��H  "          @�{�fff�j�H��  ��=qCa� �fff�$z��I�����CWz�                                    Bxc��  �          @�ff�p  �C�
��z���33C[8R�p  ��
�3�
���CP�=                                    Bxc��  T          @������ÿ�{��\��G�CL�����ÿO\)�(Q���\C?aH                                    Bxc�!:  �          @�z���=q�)����Q�����CU
��=q��
=�*=q���CJn                                    Bxc�/�  �          @��������.�R��p����CUY����Ϳ޸R�.�R���CJ��                                    Bxc�>�  �          @���|���J�H���
��z�CZ�q�|���{�.�R����CQT{                                    Bxc�M,  �          @�(���  �AG���
=��(�CY���  ���A��	33CM��                                    Bxc�[�  �          @����p��2�\�����
=CU���p���p��8�����CJz�                                    Bxc�jx  �          @����������H��\)CP�3���׿���0����=qCD��                                    Bxc�y  
Z          @�{��p��ff�   ���CQc���p����
�3�
�(�CE{                                    Bxc���  "          @�{��G��#�
�   ��CTp���G���p��9����CH#�                                    Bxc��j  �          @�33�z�H�33������(�CO�
�z�H����(Q���HCB��                                    Bxc��  T          @����|(��\)���H����CTE�|(�����&ff��=qCIQ�                                    Bxc���  
�          @�p��~{�?\)���\�f=qCX���~{���=q��z�CP��                                    Bxc��\  �          @���Vff�L�Ϳ�{�W�C_�Vff�(��ff���HCX)                                    Bxc��  
�          @����Q���������lz�CX8R�Q녿�p���p�����CO�\                                    Bxc�ߨ  
�          @��\�l�Ϳ}p������CB��l�;�녿�p���
=C:@                                     Bxc��N  
�          @���
=q@  �!G����C��
=q@E������G�B��f                                    Bxc���  "          @�  ��=q@&ff����
B�8R��=q@XQ쿏\)���\B���                                    Bxc��  �          @hQ��G�@z�������Ch��G�@2�\������\)B��
                                    Bxc�@  �          @b�\�#33?����{�33C:��#33?�33��=q����CO\                                    Bxc�(�  "          @��
�s�
?:�H�;��p�C)(��s�
?���
=��RC��                                    Bxc�7�  "          @��H�n�R?c�
� �����C&�f�n�R?��Ϳ���C��                                    Bxc�F2  T          @����s�
?s33������C&\�s�
?���
=���C��                                    Bxc�T�  �          @��\�p��?+��!G��C)��p��?�33�G���\)CB�                                    Bxc�c~  
�          @���o\)>�z��/\)���C/���o\)?�33����Cu�                                    Bxc�r$  �          @��
�j=q>aG��L���$\)C0���j=q?��
�4z����C^�                                    Bxc���  �          @��
�mp�>�  �HQ�� \)C00��mp�?\�0  �	C�                                     Bxc��p  T          @���z=q�#�
�0����HC6T{�z=q?}p��%��z�C%�R                                    Bxc��  
�          @�33������U���RC;����?Y���QG��z�C(W
                                    Bxc���  �          @��
������R�Fff���CD����>���U�ffC1��                                    Bxc��b  
�          @�z��l(��\)�7
=�=qCSQ��l(��^�R�a��,  CAB�                                    Bxc��  �          @�33�����  �޸R��Q�Cr� ����g��b�\�  Ck�                                    Bxc�خ  
�          @�ff�Q���Q쿣�
�O\)Ct�)�Q������N{�	�RCo�\                                    Bxc��T  
�          @�p��,����
=��z��>{Cq���,������33���Cn��                                    Bxc���  �          @����8Q����\���<z�Co(��8Q��{��C33�   Ci�R                                    Bxc��  
�          @��\�<(���Q��z���=qCnO\�<(��j�H�]p��\)CgL�                                    Bxc�F  
�          @�33�8Q����H��  �m�Co33�8Q��s�
�U�Ch޸                                    Bxc�!�  "          @�  �N�R���\��Q��@(�Cj� �N�R�l���<����{Cd��                                    Bxc�0�  
�          @�Q��p����Q��33��\)Cb���p���@  �H���z�CZ�{                                    Bxc�?8  �          @��R���I���\)��G�CY����(��Tz����CMB�                                    Bxc�M�  �          @�{�����H�ÿ��R���RCX\)������
�E�z�CM��                                    Bxc�\�  
�          @�{��  �0  �������CSs3��  ��33�Dz���RCH�                                    Bxc�k*  
(          @��R��z��{�,����\CIǮ��z�(��N{�	�HC;u�                                    Bxc�y�  �          @����p������0�����
CF�\��p����
�I����C7�3                                    Bxc��v  
�          @��H���H�333�p���=qC;Ǯ���H>�  �33�\C1#�                                    Bxc��  "          @�(���{���
=��=qC9���{>�(��
=���RC/E                                    Bxc���  �          @����p���
=��R�иRCJs3��p��E��B�\��HC=ff                                    Bxc��h  �          @��H���
��Q��0����{CE^����
�W
=�G
=��
C6�f                                    Bxc��  "          @������=u������C3B�����?�G��	�����
C()                                    Bxc�Ѵ  
�          @�ff���?��������  C'T{���?�=q���
��Q�C޸                                    Bxc��Z  �          @�G����\>��'��홚C2n���\?�z��ff��G�C%��                                    Bxc��   
�          @�33���?k������(�C(�\���?�ff��=q����C޸                                    Bxc���  �          @�����
=?p���(����C(���
=?�=q��{��C��                                    Bxc�L  
�          @������>���Q���G�C/#�����?����G����C$�\                                    Bxc��  �          @����@�Ϳ��K�C���@�R�L���Q�C�q                                    Bxc�)�  "          @��H����?�p���
=��z�CE����@�R�B�\�
=C�R                                    Bxc�8>  "          @�(�����?�������ӮC�����@/\)��
=�v=qC�                                    Bxc�F�  "          @����
=@
=��\���C��
=@0  ����4(�Cu�                                    Bxc�U�  �          @�{��@
=�����C���@@  �}p��'�
C�                                    Bxc�d0  �          @��
��?�{�\��z�C$����?��Ϳ\(����Cn                                    Bxc�r�  
�          @�{���?z�H�	����\)C(�����?޸R�˅����C @                                     Bxc��|  T          @�  ����?.{�ff��p�C,Q�����?�����z�����C#��                                    Bxc��"  "          @�G���z�>��H��(���G�C.����z�?�p���\)��z�C&��                                    Bxc���  �          @�����z�?(��������C-.��z�?�=q�Ǯ����C%}q                                    Bxc��n  "          @�����33?�R����ffC-���33?�{��\)���HC%�                                    Bxc��  "          @�=q���R>�  ��
=����C133���R?}p���
=���RC)=q                                    Bxc�ʺ  T          @�����z�>��   ��  C2����z�?c�
������C*!H                                    Bxc��`  �          @�  ���
�
=q�5��ffC9�����
�aG��\(��G�C6W
                                    Bxc��  �          @����׿�=q�����\C?�=���׿J=q�fff�C<�                                     Bxc���  "          @�p�������\�8Q��CA�f������Ϳ(������C?�H                                    Bxc�R  �          @�(�����W
=�Y�����C=������G���{�>�\C8�)                                    Bxc��  �          @�ff��G����!G�����C@n��G��Tzῃ�
�.�RC<�                                    Bxc�"�  �          @�������G��L�Ϳz�CA�=�����녿
=q��
=C@O\                                    Bxc�1D  
�          @�  ����
=�8Q����CE������\����\��CAǮ                                    Bxc�?�  
�          @��\���R�
=q����5�CK����R���ÿ�=q��\)CE��                                    Bxc�N�  "          @��������\��  CKL���������p����HCC�                                    Bxc�]6  T          @�ff��z���ÿ\���CHff��zῌ���ff���C@��                                    Bxc�k�  �          @�z������У׿=p����CEٚ������(�����b�\CA�\                                    Bxc�z�  T          @�����{��녾�ff��G�CC���{��{�s33�!�C@�                                    Bxc��(  "          @��
��Q쿽p���{�?�
CDu���Q�s33�˅���\C>�                                    Bxc���  T          @�(����\�ٙ�=L��?
=qCFxR���\�˅�����CE\)                                    Bxc��t  �          @�{����xQ�?�\)Alz�C>� �����?fffA=qCC�                                    Bxc��  
�          @�\)����.{������CS�������\��=q��p�COxR                                    Bxc���  "          @�
=��ff�*�H���
�aG�CR�q��ff�ff����h��CO޸                                    Bxc��f  
�          @�����H���
?}p�A&�HCGE���H�G�>�  @%�CI��                                    Bxc��  �          @����  �?G�A�CJ�H��  ��R�u�&ffCL                                      Bxc��  "          @�\)����{�Y����HCO33��녿�(���(���ffCJE                                    Bxc��X  
(          @�  ��G��)���   ��ffCP�f��G���׿��H�u�CMG�                                    Bxc��  "          @������H�C33�\�~{CU�{���H�+���  �|  CRJ=                                    Bxc��  T          @�\)�����@  �Ǯ��33CUc������(Q쿾�R�|z�CR�                                    Bxc�*J  "          @�Q����\�@�׾8Q��CUJ=���\�.{����Yp�CR��                                    Bxc�8�  �          @�����ff�7�����{CSff��ff�)�������8��CQh�                                    Bxc�G�  �          @�  ��  ��>�\)@=p�CL����  �{�
=���
CK�3                                    Bxc�V<  
�          @�  ��{���>�p�@tz�CM�)��{�
=������CM��                                    Bxc�d�  T          @����Q��
=q?�@�ffCKE��Q��p���=q�6ffCK��                                    Bxc�s�  T          @���  ��(�?B�\A�CIz���  �
=���
�k�CJ�H                                    Bxc��.  T          @��\���
�s33?�G�A/33C>�=���
��G�?z�@�=qCA��                                    Bxc���  
�          @�\)���
��G�?h��A Q�C8����
�@  ?.{@�C<J=                                    Bxc��z  
�          @�p�������(�?��
A7�C8ٚ�����G�?L��A{C<                                    Bxc��   "          @�z���{�fff?��A;\)C>E��{��(�?#�
@���CA��                                    Bxc���  �          @�G���(��˅?��HAT��CF���(���
=?
=q@��CI�=                                    Bxc��l  �          @�����
=��ff?��
A`��CB�f��
=��
=?8Q�@�=qCF�                                    Bxc��  �          @�
=���
��z�?�
=A\)CA^����
����?h��A ��CF33                                    Bxc��  �          @�����G����?�Az�\CF����G��33?8Q�@�z�CK!H                                    Bxc��^  
�          @������ÿ��
?У�A��CE�3�����G�?uA'\)CJ�
                                    Bxc�  
�          @�Q�������H?�(�A��CH#�����{?z�HA,z�CMu�                                    Bxc��  �          @������R��
?��\A^=qCK�)���R�Q�>�G�@�\)CN�\                                    Bxc�#P  
�          @�{��ff��=q?ٙ�A�CC�\��ff��{?���AEG�CI��                                    Bxc�1�  �          @������\��\)?��A���CD�f���\���H?�  AbffCK!H                                    Bxc�@�  �          @�Q���  ��?\)@���CD�f��  ���
<#�
=�\)CE��                                    Bxc�OB  T          @�Q�������u�%CI������ٙ��h��� ��CGT{                                    Bxc�]�  �          @��\���   �aG���CJ
=�����
�k��\)CG�f                                    Bxc�l�  "          @�  ��Q����#�
��CK}q��Q��ff�����hQ�CGǮ                                    Bxc�{4  "          @�33���
�
�H�8Q��(�CL����
���H�s33�$(�CI�                                    Bxc���  T          @�\)��z���\?5@�=qCNQ���z�����#�
��p�COQ�                                    Bxc���  
Z          @�
=��  �p�?��A@��COh���  �-p�>B�\?��HCQ�3                                    Bxc��&  
�          @�  ��p��p�?��
A��CO��p��5?��@�Q�CSO\                                    Bxc���  "          @�33�������?���AF�\CN#������*�H>�=q@/\)CP��                                    Bxc��r  "          @�\)����\)>�@��CK������׾�{�Z�HCKY�                                    Bxc��  "          @�G�����5�?:�H@���CQ�=����9�����
�H��CR33                                    Bxc��  
�          @����Q��0  ?�\@��HCPǮ��Q��0  �   ���RCP��                                    Bxc��d  
�          @������%�>�p�@n�RCN������#33�\)��(�CN��                                    Bxc��
  "          @�
=��33�#�
=�G�?��CN����33�(��L�����CM�=                                    Bxc��  "          @������)��>k�@�CO�=�����#�
�5��
=CO                                      Bxc�V  "          @��R������ͽ��Ϳ��
CMs3������׿u�ffCK��                                    Bxc�*�  
�          @������\�p����
�G�CM�
���\�녿n{�Q�CL&f                                    Bxc�9�  �          @�\)��ff�ff���
�\(�CM\)��ff�
�H�fff��RCK��                                    Bxc�HH  
�          @����33�   �k���CO@ ��33��׿�=q�7
=CM�                                    Bxc�V�  �          @�G���p��#33�#�
��p�COff��p��Q�k����CM��                                    Bxc�e�  �          @��������R�.{��(�CK�������\�n{�
=CI�                                    Bxc�t:  T          @�����{�W�?:�H@�\)CY:���{�Z�H����p�CY�
                                    Bxc���  �          @�����=q�9��>��H@�{CS���=q�8�ÿ�����CR��                                    Bxc���  .          @�33����!G���=q�/\)CN����G���\)�8��CLu�                                    Bxc��,  H          @��H��p��'
=�L�Ϳ�\CO���p��(��s33�Q�CNO\                                    Bxc���  �          @�����\�0  ?0��@��HCQ�����\�4zᾙ���A�CRE                                    Bxc��x  �          @�33����0  >�ff@�33CQ}q����.�R�����HCQ^�                                    Bxc��  
�          @�����\�-p�>�G�@��\CQE���\�,�Ϳ�\����CQ(�                                    Bxc���  �          @�33�����*=q��z��?\)CPp����������
=�C\)CN
=                                    Bxc��j  �          @�����H�(�������
CKO\���H� �׿c�
�\)CI�\                                    Bxc��  �          @�=q�����<��
>L��CM�������\�O\)�{CL�                                    Bxc��  �          @������\)�u�\)CKp�����O\)���CJ                                      Bxc�\  �          @��
��
=�zᾣ�
�R�\CI���
=����G��$��CGs3                                    Bxc�$  �          @��
��=q��\�����Tz�CFn��=q�Ǯ�h���z�CDT{                                    Bxc�2�  �          @�z����
��
��Q��l(�CLO\���
��
��\)�7�
CI޸                                    Bxc�AN  �          @�����
����ff����CH�����
��33����2�HCE�f                                    Bxc�O�  "          @�
=�����녾����(�CI��������G���\)�=G�CG(�                                    Bxc�^�  �          @�{������
=�
=q���RCH޸�����У׿�z��DQ�CE�                                    Bxc�m@  "          @��R��(���
=�G���CF!H��(����ÿ�ff�\Q�CBp�                                    Bxc�{�  �          @�p���
=����
=���CK���
=��{��{�<��CH}q                                    Bxc���  �          @�����ff����ffCMp�����\���\�V�\CJu�                                    Bxc��2  "          @������R��Ϳ�\���CN@ ���R��ÿ�ff�X(�CKL�                                    Bxc���  `          @������R���#�
���
CM�����R�녿���hz�CJ:�                                    Bxc��~  H          @�����
��
=�aG��=qCH�����
���
��(��v{CD�f                                    Bxc��$  �          @�z���Q����h����CF��Q쿰�׿����n�HCB                                    Bxc���  T          @�33�����������
�)��CCJ=�������
�����pQ�C>�q                                    Bxc��p  �          @�  ��\)��ff��=q�5��CA����\)�^�R�����t��C=xR                                    Bxc��  �          @��\��Q쿽p������:=qCC�3��Q쿃�
��ff���HC?�                                    Bxc���  T          @��H������H��G��P  CC�=����xQ��z����RC>s3                                    Bxc�b  �          @����  �Ǯ��
=�C
=CD�
��  ����У���33C?�                                     Bxc�  �          @�33��
=�У׿�
=�B�HCEO\��
=��33��33���HC@u�                                    Bxc�+�  �          @��\������zῬ���_�
CH��������\)�����
=CC#�                                    Bxc�:T  �          @�G���
=��
��ff�Xz�CJ� ��
=���
�����  CE{                                    Bxc�H�  �          @������׿���
=�o�CG�=���׿�  ��
=��{CA��                                    Bxc�W�  T          @�  ���׿�ff�Ǯ��ffCE&f���׿u��p���G�C>�
                                    Bxc�fF  �          @�{���ÿ�
=��
=��Q�CA(����ÿ녿�(���=qC:z�                                    Bxc�t�  �          @�\)���ÿfff��(����\C>&f���þaG��
=q��G�C6�                                     Bxc���  T          @�ff���ÿE�������G�C<�3���ý������ffC50�                                    Bxc��8  �          @���Q�Q녿����Q�C=T{��Q�8Q���\��G�C6
=                                    Bxc���  �          @������׿��ÿ�p���=qCF(����׿Y���Q���
=C>)                                    Bxc���  T          @�����������ĸRCA������Ǯ�!G����HC8��                                    Bxc��*  �          @���=q�����p����\CA0���=q��p������p�C8^�                                    Bxc���  �          @��\����\)�ff�ѮC:�=���>W
=�=q��33C1u�                                    Bxc��v  
�          @�Q�����aG������
C6���?�����֣�C-5�                                    Bxc��  �          @���33�#�
����=qC4
��33?=p��  ��z�C*�
                                    Bxc���  �          @���Q��l�Ϳ�z��~{Cdk��Q��Dz��   ��RC_�                                    Bxc�h  �          @��Vff�hQ���
���CcL��Vff�=p��%���C]�                                    Bxc�  �          @�33�k��E���(���\)C[�R�k�����&ff����CU�                                    Bxc�$�  �          @����n{�E���p�����C[���n{����'
=��Q�CT��                                    Bxc�3Z  T          @�G��0  �]p�����h  Cg}q�0  �;��
�H��z�Cb�=                                    Bxc�B   "          @�
=?E����\?��\A<(�C���?E�����
=q���C�z�                                    Bxc�P�  T          @��R?������?���Ak33C�w
?������\��=q�=p�C�B�                                    Bxc�_L  "          @�\)��z������{��(�C{T{��z��vff��{��p�Cy�3                                    Bxc�m�  �          @�\)�   �u����
�U��CrY��   �Tz��
=q���Cn��                                    Bxc�|�  �          @�=q�vff?0���@  �33C)�=�vff?˅�(���C��                                    Bxc��>  �          @�����>Ǯ�E�(�C.����?����3�
�p�C"!H                                    Bxc���  �          @�=q��=q?�(��Y����C !H��=q@(��1�����C�                                    Bxc���  �          @�(��mp�@!��Vff��C�3�mp�@\(���H��=qC	33                                    Bxc��0  �          @�33�l(�@z��aG��C���l(�@R�\�)����G�C
L�                                    Bxc���  �          @���s�
@��\(���C��s�
@S�
�#�
��33C                                    Bxc��|  �          @��
�o\)@  �aG��Q�C���o\)@Mp��*�H��
=CY�                                    Bxc��"  �          @��\�w
=@�Z�H��RC�H�w
=@A��(Q���\)C�                                    Bxc���  �          @�=q�z=q@
=�W
=��\C���z=q@AG��$z��ۮCE                                    Bxc� n  �          @��\�~�R?���Z=q�  C�R�~�R@4z��+���ffC��                                    Bxc�  �          @��H�l(�@!G��W
=��C��l(�@Z�H�(���\)C	=q                                    Bxc��  �          @�Q��j�H@=q�U����C�R�j�H@S33�����G�C
\                                    Bxc�,`  �          @�
=�s33@���P  �(�C���s33@Dz�����ԸRC�                                    Bxc�;  �          @���xQ�@ff�XQ����C�
�xQ�@@���&ff�ޣ�C.                                    Bxc�I�  �          @�(�����?����R�\���C޸����@�R�+���C��                                    Bxc�XR  T          @�=q��?�{�HQ���C�)��@{� ������C\)                                    Bxc�f�  "          @���\)@5�p���ffC���\)@W
=���\�Z�\C�
                                    Bxc�u�  �          @���|��@L(�����ffC)�|��@j=q��ff�/�C	.                                    Bxc��D  �          @���  @C�
����\)C�=��  @a녿��
�/\)C
�
                                    Bxc���  �          @��
���
@-p��ff��ffC�����
@Mp����H�P(�C
                                    Bxc���  �          @�����@�������G�C�)����@=p����H�yG�C�                                     Bxc��6  �          @�=q���
@�G�����C�H���
@*�H�Ǯ��=qC                                      Bxc���  
Z          @�Q����@(���R��33C\���@C�
��z����HC+�                                    Bxc�͂  T          @��R����@1G�����p�Cu�����@W
=��G�����C&f                                    Bxc��(  �          @��H����@?\)��R����CaH����@e���  �y��C
O\                                    Bxc���  �          @�z���{@6ff�p�����C���{@\(��\�{33C�)                                    Bxc��t  T          @�33��33@G��$z���
=Cn��33@:�H��ff���
C&f                                    Bxc�  �          @������@�7
=���C�q����@5��������C��                                    Bxc��  �          @��H��{?�G��8������Cc���{@!G������G�Cs3                                    Bxc�%f  �          @������\?��(����C�����\@
=�33���C                                    Bxc�4  �          @�=q���\?�\�(Q���C�f���\@���G����
C޸                                    Bxc�B�  "          @�����
=?����9�����
C"
=��
=@{����33C�
                                    Bxc�QX  "          @�Q����R?�z��:=q��G�C"p����R@�����ͅC��                                    Bxc�_�  
�          @��R��33?�G��;����C �\��33@�\�Q�����CG�                                    Bxc�n�  �          @������?��H�<(����C!=q����@�R�����
=C�=                                    Bxc�}J  �          @��H����?�=q�:�H��RC"�3����@ff����=qC�                                    Bxc���  �          @�����33?����3�
� (�C%n��33?���Q���33C��                                    Bxc���  �          @�����?�{�9���=qC%�����?���{���
C��                                    Bxc��<  �          @���  ?+��:�H���C+����  ?�(��'���C!�                                    Bxc���  �          @����>�p��@  �=qC/E��?�Q��1����
C$��                                    Bxc�ƈ  �          @��R��=q>����8����C/�{��=q?����,(���{C%Y�                                    Bxc��.  �          @�z����    �1��(�C3�����?G��*�H����C)�                                    Bxc���  �          @����{>���0  ��{C2���{?h���&ff��RC(h�                                    Bxc��z  
(          @����>�(��/\)���C.�3���?�
=� �����C%�{                                    Bxc�   "          @�����(�>�ff�/\)���
C.u���(�?�Q�� ����\)C%�
                                    Bxc��  
�          @������R>�Q��@����RC/Y����R?�
=�2�\���C%.                                    Bxc�l  T          @����\<#�
�5��z�C3�f���\?J=q�.{��G�C)�R                                    Bxc�-  T          @�G����R=����0���C2�H���R?Y���'����C(�=                                    Bxc�;�  
(          @�����Q���6ff���C:5���Q�>�{�7���\C/}q                                    Bxc�J^  T          @�����G��@  �<���z�C=�)��G�=�Q��B�\�Q�C2�\                                    Bxc�Y  T          @�p���  >�  �8Q��=qC0�
��  ?��\�,����RC'0�                                    Bxc�g�  �          @�Q���p�?�  �.�R���C'�R��p�?ٙ��ff�ʣ�C �                                    Bxc�vP  
�          @��
��33>�
=�(����p�C.p���33?�\)����G�C%��                                    Bxc���  "          @�(���{>aG��"�\���C1(���{?fff�Q����C(��                                    Bxc���  �          @�z�����?G��
=��(�C*������?���z���Q�C#��                                    Bxc��B  �          @�����p�?z�H��
��  C(����p�?��ÿ��H���\C"Q�                                    Bxc���  "          @�����
=?5�33��z�C+޸��
=?�ff�G����C%W
                                    Bxc���  �          @�
=��p�?(����\��ffC,ff��p�?�  �����RC%�=                                    Bxc��4  "          @�
=���>���   ���
C.�����?}p���ff��{C(��                                    Bxc���  "          @�p����
>\)�޸R���
C2n���
?��������HC-W
                                    Bxc��  
�          @�����<��
�@����C3�3����?Q��9��� �C)��                                    Bxc��&  z          @��H���H����������C7����H>���������C0��                                    Bxc��  
�          @�Q���  ��  �  �ʸRC7���  >�33�\)��p�C/Ǯ                                    Bxc�r  
N          @����=q    �%���\)C3���=q?.{��R��(�C+}q                                    Bxc�&  �          @�\)���\?z��3�
�C,k����\?���#�
��C#^�                                    Bxc�4�  �          @���  ?���9���Q�C,�\��  ?�ff�)�����RC#
=                                    Bxc�Cd  �          @�
=����?
=q�<���	��C,�)����?���-p���=qC#5�                                    Bxc�R
  "          @�������=��
�&ff���C2�H����?B�\�\)���C)�R                                    Bxc�`�  
�          @�{���\����.{���C<\)���\>��1��	�C233                                    Bxc�oV  �          @�(��u��ff�3�
��CCY��u��z��?\)�Q�C8J=                                    Bxc�}�  �          @�33�l(�����7��{CGE�l(��   �Fff�=qC;�                                    Bxc���  �          @�G��b�\�����9����CIB��b�\�z��J=q�%z�C=T{                                    Bxc��H  T          @���dzῦff�B�\�z�CH�dz���QG��(�HC;�
                                    Bxc���  �          @�=q�a녿�=q�>{��RCH���a녿��Mp��'�C<�                                     Bxc���  �          @��H�R�\� ���=q���CYff�R�\���>�R���CP�)                                    Bxc��:  
�          @����p��i���n{�?�Cl
=�p��R�\������Ci=q                                    Bxc���  �          @���������
=��  C}𤿙���|(���(���G�C}                                      Bxc��  �          @���(����33=�Q�?��HC�t{�(���\)�u�U�C�T{                                    Bxc��,  �          @�(��}p���  �\)���RC�3�}p��u��33���
C��                                    Bxc��  
�          @������\)�����C~�ÿ���o\)���H��{C}                                    Bxc�x  	�          @��R����}p��\)��C{������l(��������Cz�3                                    Bxc�  	�          @}p��Tz��j�H��  ����C���Tz��N�R�Q��
=C��                                    Bxc�-�  �          @��׿��
�j�H����z�CwaH���
�Z�H��Q����
Cu��                                    Bxc�<j  T          @��������Q쾅��aG�C{Y������s�
��  ���Czz�                                    Bxc�K  "          @��R��Q��\)�\)���C}uÿ�Q��n�R�����C|Y�                                    Bxc�Y�  �          @�(��У���G��.{��\Cx��У��w���
=�y�Cw(�                                    Bxc�h\  �          @�p�������H�k��?\)Cx�����z=q���R��=qCw.                                    Bxc�w  �          @�{��(����\���
���Cw
��(��w�������(�Cv�                                    Bxc���  
�          @�=q�����\���\Cv�����|�Ϳ�33���Cup�                                    Bxc��N  a          @�녿�  ��{�\��\)CwB���  �}p��������Cv!H                                    Bxc���  y          @��\��\)���׾���Mp�Cy0���\)���\��ff��CxJ=                                    Bxc���  �          @��
�33������R�u�Csu��33�z=q������CrJ=                                    Bxc��@  
�          @��\��ff���?(��@��Cz녿�ff��Q����  C{�                                    Bxc���  �          @�ff������?�G�A��\C�)������>�Q�@�z�C�#�                                    Bxc�݌  �          @�
=��(���p�>��@��C~�)��(���(��.{�
=C~�                                     Bxc��2  �          @�\)������R�333�  C}LͿ�������G����C|+�                                    Bxc���  �          @��
��Q���(����H��p�Cz� ��Q��i���=q��ffCxxR                                    Bxc�	~  a          @�{��(��mp��{��{CxT{��(��@  �R�\�2{Cs޸                                    Bxc�$  �          @��\����@���e��@G�CyB������
��ff�r�Cq\                                    Bxc�&�  T          @��R��\�p����\�\��CgJ=��\�����p�CUaH                                    Bxc�5p  �          @����   ��
�r�\�S(�Ca�H�   ������p��u(�CP�                                    Bxc�D  T          @�>k��j�H>�@C��>k��fff�@  �:�HC��                                    Bxc�R�  
�          @�G�?�Q����
@��A�  C��?�Q���Q�?���AS�C�b�                                    Bxc�ab  �          @��\?�����?�33A��
C���?�����  ?J=qA  C�*=                                    Bxc�p  �          @�
=?�����@G�A��HC�P�?����
=?}p�A9�C�l�                                    Bxc�~�  
�          @��@G���{@A�G�C�AH@G����?z�HA*ffC�k�                                    Bxc��T  
�          @���?��
��z�?���A�  C�� ?��
��z�?�@�\)C�Ff                                    Bxc���  T          @�{?z�H��33?�z�A�=qC�^�?z�H���>�33@���C��                                    Bxc���  �          @��
?s33��
=?�{A���C�Q�?s33��\)?�@�z�C��
                                    Bxc��F  �          @�ff?xQ���G�?�=qA�Q�C�^�?xQ�����?
=q@ӅC��                                    Bxc���  �          @��
>���(�?�Q�A��C�f>�����?�R@�C�ٚ                                    Bxc�֒  �          @�=����?L��A(�C�� =����;����C���                                    Bxc��8  T          @��R�8Q���p�?�R@���C���8Q�����&ff��33C��                                    Bxc���  �          @��ýu��Q�>���@�(�C���u���R�c�
��C���                                    Bxc��  �          @�{��Q����    ���
C�)��Q����׿�  �T��C�                                    Bxc�*  
�          @��R�.{���;�z��N�RC�AH�.{��{��(���33C��                                    Bxc��  �          @�{�xQ���=q�L���
�HC���xQ���(����s
=C���                                    Bxc�.v  �          @�(���p����R���
�W�C�W
��p���  ��  ��C��                                    Bxc�=  �          @�33�Ǯ�����  �,z�C|�׿Ǯ��(��
=���\C{aH                                    Bxc�K�  �          @�(��s33����fff�=p�C��=�s33�vff������
=C��                                    Bxc�Zh  �          @�  �c�
�u��R�ffC�xR�c�
�fff��(���33C��                                    Bxc�i  �          @~�R�!G��l�Ϳ��
��=qC�,Ϳ!G��S�
�����C��                                     Bxc�w�  T          @[��G��/\)��=q�  C~ͿG��  ����4Cz�)                                    Bxc��Z  �          @[�    �XQ콣�
��33C���    �Q녿W
=�f�\C���                                    Bxc��   �          @~{�B�\�s33�s33�`  C��{�B�\�_\)���
�י�C�t{                                    Bxc���  
�          @xQ�<#�
�u�������RC�
<#�
�i����  ���C�R                                    Bxc��L  �          @~{>���z�H>�=q@}p�C�>���xQ�(��C��                                    Bxc���  T          @���?�  �HQ�?�\)A�Q�C���?�  �^{?���A�{C��                                    Bxc�Ϙ  �          @��\?��H�@C�
B4{C��3?��H�=p�@p�B
��C��)                                    Bxc��>  T          @z=q?�����@5�B:p�C�Z�?�����H@
=B�RC�p�                                    Bxc���  �          @n{?�=q��Q�@%�B0\)C��?�=q�p�@B	�C�J=                                    Bxc���  �          @i��@   ���@(Q�B8�C�AH@   ��@{Bp�C���                                    Bxc�
0  �          @aG�@p��\@*�HBH
=C�)@p��u@!G�B9\)C�E                                    Bxc��  �          @Z�H@{�&ff@Q�B7��C�� @{��33@�B$��C�B�                                    Bxc�'|  �          @aG�@���\)@(�B0G�C��3@���\)@�B��C�K�                                    Bxc�6"  �          @aG�@33��{@!G�B6��C�:�@33��\)@
=qBC���                                    Bxc�D�  �          @c�
@����@#33B7�C��{@���\)@(�Bz�C��                                    Bxc�Sn  �          @g�@Q쿫�@&ffB7��C��
@Q��\)@\)B�C�e                                    Bxc�b  �          @k�@Q쿊=q@1G�BE{C���@Q��33@{B*{C�                                    Bxc�p�  �          @g�?�
=�
=@�
B   C��
?�
=�333?�  A�RC��                                     Bxc�`  �          @c�
?�{�(�@{B�RC���?�{�7
=?��A�
=C���                                    Bxc��  �          @h��?�G��33@=qB'33C�U�?�G��!G�?�A��
C�xR                                    Bxc���  T          @j=q?�p��Q�@
�HB��C�?�p��2�\?�{A�
=C��=                                    Bxc��R  �          @qG�?�  ��R@��B33C���?�  �<(�?�A�{C��                                    Bxc���  T          @p  ?�  �!G�@�B�HC�g�?�  �=p�?޸RA�p�C�w
                                    Bxc�Ȟ  
�          @|(�?�G��*�H@p�B
=C��=?�G��HQ�?�A�
=C��                                    Bxc��D  T          @���?�ff�1G�@!G�B�HC���?�ff�O\)?��AܸRC���                                    Bxc���  �          @l(�?�(�� ��@B=qC�1�?�(��<��?�G�A��\C�C�                                    Bxc���  �          @w�?�G��Dz�?�p�A���C�1�?�G��Z=q?��A�  C�&f                                    Bxc�6  T          @s33?�G��3�
@�B��C�
?�G��K�?�33A��C��\                                    Bxc��  "          @vff?h���H��?���A��RC��?h���^�R?�  A�G�C�K�                                    Bxc� �  T          @z�H>k��Z=q?�A�G�C���>k��n�R?�A��HC���                                    Bxc�/(  T          @s�
?
=q�Q�?�{A�G�C��3?
=q�fff?���A�{C�K�                                    Bxc�=�  T          @|(�?L���^{?�
=A��
C�u�?L���o\)?k�AW�
C�H                                    Bxc�Lt  T          @���?=p��j�H?�ffA���C���?=p��z=q?@  A*�HC�Q�                                    Bxc�[  �          @��׽u�z�H?O\)A;�C��=�u��Q켣�
�aG�C���                                    Bxc�i�  �          @�녿�z��s�
    =#�
C}���z��n�R�G��5C|�                                    Bxc�xf  �          @�Q�>�33�p  ?�G�A�p�C���>�33�{�>�@ҏ\C��                                    Bxc��  T          @�Q�>L���r�\?�  A�C�� >L���~{>�G�@��
C�o\                                    Bxc���  "          @�Q쾨����z�?fffAC\)C��
������\)<�>��C��                                    Bxc��X  "          @����p�����?E�A"{C��\��p����H�����C��R                                    Bxc���  �          @��H�L����ff?�ffA`��C��)�L�����\>.{@�\C��f                                    Bxc���  T          @��
��33���?xQ�AL��C�� ��33��33=��
?��C��\                                    Bxc��J  �          @��������p�?���A�(�C�T{�������?   @љ�C�\)                                    Bxc���  T          @��H<�����?��RA�p�C�,�<����>\@�  C�+�                                    Bxc��  �          @��׾�33���
?��Ac
=C��;�33���>B�\@%�C���                                    Bxc��<  �          @�>����Q�?�{Aw�C��>������>�\)@vffC��                                    Bxc�
�  T          @��R=L����=q?��
Ad  C�XR=L����ff>B�\@)��C�U�                                    Bxc��  
�          @���������?fffADz�C�0��������R=u?B�\C�AH                                    Bxc�(.  T          @��R�\)���\?W
=A9G�C�{�\)��p�<#�
>\)C�(�                                    Bxc�6�  T          @��Ϳ=p���  ?L��A2{C��=p����\�#�
�L��C��)                                    Bxc�Ez  T          @��
�333��{?z�HAO
=C�@ �333����=�?�{C�^�                                    Bxc�T   �          @�(��#�
���R?�  AS\)C��\�#�
���\>\)?��C���                                    Bxc�b�  "          @�Q���H���?8Q�Az�C��)���H��\)������C���                                    Bxc�ql  �          @��z���  ?���Ae��C�  �z���(�>u@C33C�=q                                    Bxc��  �          @������R��p�?��AS
=C�����R��G�>��?�C�\                                    Bxc���  �          @��ýL����
=?(��AC��{�L����Q�u�?\)C���                                    Bxc��^  �          @��þW
=��z�?�G�AO�C��f�W
=��Q�>�?�C��\                                    Bxc��  �          @�z᾽p����?�{A]��C�� ��p����
>W
=@%C���                                    Bxc���  T          @�G�������{?h��A.�\C�׾������ü�����C��                                    Bxc��P  �          @�z�k���{?���A�ffC�� �k����>�(�@��\C��                                    Bxc���  �          @�z�.{����?�\)A���C�� �.{���\>��H@��
C��                                    Bxc��  T          @�녿����z�?�\@θRC�\)������;�33��C�b�                                    Bxc��B  �          @�녿�  ��(����Ϳ��RC~����  ���׿�  �@(�C~��                                    Bxc��  �          @�p����\��\)�u�@  C~:ῢ�\��33��{�[\)C}Ǯ                                    Bxc��  T          @�  �c�
��33�&ff���
C���c�
��z�\��(�C�C�                                    Bxc�!4  �          @��׿}p����H�B�\�=qC��=�}p�����У����C���                                    Bxc�/�  �          @�\)�}p����׿p���7
=C��3�}p���Q������
C�t{                                    Bxc�>�  �          @�{��33��ff�L���C�(���33���R��
=���\C�)                                    Bxc�M&  T          @��xQ�����>�p�@�=qC�AH�xQ���  �����
=C�<)                                    Bxc�[�  T          @���h������?333A=qC��ÿh����=q�k��)��C���                                    Bxc�jr  T          @��\�Tz���(�?�ffAH��C�箿Tz����>��?�\C�f                                    Bxc�y  T          @�33�E���{?fffA*�RC�XR�E����׼#�
��C�l�                                    Bxc���  �          @�p������H?+�@��
C�⏿���(���  �>{C���                                    Bxc��d  �          @�녿(����R?#�
@��RC�XR�(���  ����E�C�^�                                    Bxc��
  �          @�  �&ff����?
=q@�{C�>��&ff���������=qC�AH                                    Bxc���  "          @�  ���
���?�=qAv�HC�R���
��
=>���@��
C�'�                                    Bxc��V  "          @�������(�?У�A�
=C��;�����33?=p�A	C���                                    Bxc���  �          @���#�
��33?��
A@(�C�;#�
���R=�G�?�p�C�3                                    Bxc�ߢ  "          @�ff������?�\)AF=qC�J=����p�>��?�
=C�Z�                                    Bxc��H  �          @�{�����G�?��\A5G�C��q�������=u?+�C��                                    Bxc���  �          @�
=�:�H��z�>�ff@�  C��:�H��(���\��33C��                                    Bxc��  "          @��þL������=�G�?�  C���L����ff�\(��
=C��H                                    Bxc�:  �          @�ff����{��Q쿂�\C�Uý����H��ff�:�HC�Q�                                    Bxc�(�  �          @�33����=q�
=q���HC�H�����(���(����
C�AH                                    Bxc�7�  �          @�p��B�\���;u�+�C��þB�\��Q쿘Q��T��C��                                    Bxc�F,  �          @�G��#�
��Q����Q�C�#׾#�
��=q���R���HC�)                                    Bxc�T�  T          @�33������=q��
=��Q�C�h������������q�C�\)                                    Bxc�cx  �          @�����
��
=���H���\C��{���
��G���
=�~{C���                                    Bxc�r  �          @���=�����33�.{��C��\=�����z�������C��{                                    Bxc���  �          @�\)��Q����
�s33�(  C����Q������\)��{C�~�                                    Bxc��j  �          @�{>L�����׿��
�e�C�%>L����ff�
�H��ffC�8R                                    Bxc��  
�          @�=q�aG���zΰ�
�`  C��f�aG����\����Ù�C���                                    Bxc���  T          @�ff�����=q��Q��J{C��������Q������z�C���                                    Bxc��\  T          @���B�\��  ��p��S�C��3�B�\��{�
=q��
=C��H                                    Bxc��  "          @��ͽ�G���G�����4z�C�lͽ�G���Q���R��p�C�e                                    Bxc�ب  �          @��þW
=��{�xQ��!��C��׾W
=��p���
=��C��{                                    Bxc��N  
(          @�G�����(������
C�.���n�R�#33�	\)C�R                                    Bxc���  
�          @�33��G���녿�Q���Q�C�>���G��n�R�	����
=C�.                                    Bxc��  �          @��W
=���ÿ����Z�\C����W
=��Q��33���C���                                    Bxc�@  T          @���>�{��Q콣�
�Q�C���>�{�����G��1��C���                                    Bxc�!�  "          @�
=>�G�����{�o\)C�p�>�G���G���G��_�C���                                    Bxc�0�  "          @�{>8Q����\��G��3�C��>8Q���=q��33���
C�
                                    Bxc�?2  �          @�  ��G���z��z���z�C�b���G���
=�/\)��Q�C�S3                                    Bxc�M�  �          @�33>u���Ϳ�G����C���>u�tz��{��G�C��                                    Bxc�\~  T          @���>u�XQ��&ff�(�C�f>u�7��J=q�>�
C�b�                                    Bxc�k$  �          @�Q�    �p  ��R���C��    �P  �G
=�.�RC��                                    Bxc�y�  
�          @�
=>��ff����n  C��f>��˅���ǮC�f                                    Bxc��p  �          @��R?����l���`Q�C�(�?��޸R��=q��C�z�                                    Bxc��  �          @��R?s33����n�R�e(�C��H?s33��G�����{C�f                                    Bxc���  �          @�{?����  �h���]{C�P�?��׿�=q�~�R���C�Ǯ                                    Bxc��b  �          @��?��
�Mp��C33�(�C��?��
�(Q��dz��N�C��                                     Bxc��  T          @�(�?����dz��*=q���C�� ?����C33�P  �3��C�!H                                    Bxc�Ѯ  
�          @x��?����J=q������HC�]q?����2�\�=q��RC��                                     Bxc��T  �          @��?���dz����ffC�J=?���K��!G��z�C�=q                                    Bxc���  �          @�Q�?�z����H�
=��33C��?�z��j=q�2�\���C�Ф                                    Bxc���  "          @��?���G��
=q��p�C�R?��fff�5����C��                                    Bxc�F  �          @��?�=q�������  C�E?�=q�qG��'
=��C���                                    Bxc��  �          @��
?�ff��G���������C�~�?�ff��������33C��3                                    Bxc�)�  T          @���?�\?+��tz��~��A��
?�\?����i���kQ�B�H                                    Bxc�88  �          @�
=@�\?��R����z��A���@�\@G����\�a��B2�
                                    Bxc�F�  �          @�=q@	��=������\�{@(Q�@	��?c�
����zp�A��H                                    Bxc�U�  T          @�ff@!G�>Ǯ���R�k  A�@!G�?�{���\�_�
A��                                    Bxc�d*  �          @�p�@�?����Vff�O
=B=q@�@��A��5�B433                                    Bxc�r�  
�          @��H>��=L����=q©�3@�(�>��?Tz�����3B(�                                    Bxc��v  T          @\)?�=q?˅�N{�\�B4(�?�=q@
=�9���>��BT�\                                    Bxc��  �          @�  ?h��?���{��B\�
?h��?��H�i���m�\B�B�                                    Bxc���  �          @s�
?+�?˅�Tz��z(�B��?+�@Q��@  �UG�B���                                    Bxc��h  �          @e=�G�@{�!G��633B��=�G�@7
=��
�
=B���                                    Bxc��  
�          @]p��#�
@ff�,���P33B��=�#�
@!��33�)(�B��                                    Bxc�ʴ  �          @(Q��?����ٙ��#��B�33��@p�������
B�\)                                    Bxc��Z  �          @0  ?��?�33��Q��)\)B:��?��?��׿�p���RBO\)                                    Bxc��   �          @��R@~{������
��{C��3@~{�����p�����C���                                    Bxc���  �          @��R@vff���׿����\)C�*=@vff��ff�   ���HC�XR                                    Bxc�L  �          @�
=@����zῺ�H���C���@���#�
��p�����C���                                    Bxc��  �          @��R@��H?Q논���G�A,  @��H?Q�=�G�?�{A*�H                                    Bxc�"�  
�          @���@���>��þ��R��ff@�\)@���>\��  �X��@��
                                    Bxc�1>  T          @��@�ff=u������?@  @�ff>���ff��33?�ff                                    Bxc�?�  "          @��\@�{�u�0����C��{@�{=u�0���{?B�\                                    Bxc�N�  
�          @��
@�\)?s33��33����AJff@�\)?}p��8Q��Q�AS�                                    Bxc�]0  �          @�=q@�p�?�  �������
AW�@�p�?�ff�k��C�
Ab�R                                    Bxc�k�  �          @�=q@��H?�����
=���A��\@��H?�  �aG��9��A�{                                    Bxc�z|  �          @�{@��R?�\)=�?�=qA�(�@��R?�=q>�33@���A�=q                                    Bxc��"  T          @��\@�ff?k���p����A;\)@�ff?xQ�W
=�'
=AD��                                    Bxc���  �          @��@�p�>\�����\)@��R@�p�>�ff��{���
@�
=                                    Bxc��n  �          @��R@�(��Ǯ�+���RC��R@�(���=q�8Q��C�P�                                    Bxc��  T          @��@��H���Ϳ#�
��(�C�� @��H��z�333�	��C�1�                                    Bxc�ú  
�          @��@�G��z�5��C�  @�G����L���'�C��3                                    Bxc��`  T          @���@����333��33�z�RC�)@����   ��  ���\C��                                     Bxc��  �          @��H@j=q�z�H��Q�����C��=@j=q�8Q��=q���
C�g�                                    Bxc��  "          @tz�@5���G���z����C��3@5��u���H��
=C���                                    Bxc��R  �          @j=q@@z�>#�
@@��B?��@@��?��A z�B=
=                                    Bxc��  
�          @j�H@^�R?^�R�#�
� ��A`��@^�R?u�����
Aw�
                                    Bxc��  �          @�33@A�@o\)?z�@�33BL33@A�@fff?�Q�Ab�\BG�R                                    Bxc�*D  �          @���@P��@]p�?
=q@θRB:�R@P��@Tz�?���ATz�B6(�                                    Bxc�8�  T          @|(�@Z�H?��?�\)A�\)A���@Z�H?�33?�=qA�A���                                    Bxc�G�  �          @��@�z�@G�@��A�
=AЏ\@�z�?�33@(�A�A���                                    Bxc�V6  T          @���@�\)@z�?�A��A���@�\)?�(�@�A�A�{                                    Bxc�d�  �          @��@�=q@ ��?�
=AuG�A��@�=q@  ?���A��Aх                                    Bxc�s�  T          @���@�=q@.�R?O\)A
{A�z�@�=q@#�
?��RAU��A�                                    Bxc��(  T          @��R@n�R@tz�E���\B6�R@n�R@x�þ���33B8�H                                    Bxc���  T          @�
=@�  @g��:�H���B(�\@�  @l(�������B*                                    Bxc��t  �          @��R@��@N{?�R@��B��@��@Dz�?���AA��B                                      Bxc��  �          @���@`��@"�\@Q�A���B�R@`��@�@1G�B(�A��                                    Bxc���  T          @���@8��?���@c33B@=qAȏ\@8��?B�\@n{BM{Aj�\                                    Bxc��f  
�          @�33@L(�?���@QG�B)�RAՅ@L(�?�ff@^�RB8{A�                                      Bxc��  T          @���@O\)@	��@1G�B��B�@O\)?�@EB!  A���                                    Bxc��  �          @�(�@P��@G�@AG�B�
A�Q�@P��?�  @S33B*��A�{                                    Bxc��X  �          @�ff@b�\@%@0��B �RB�H@b�\@
=@I��B\)A�ff                                    Bxc��  �          @�@B�\?˅@�{BJ�RA܏\@B�\?c�
@�z�BX�RA�=q                                    Bxc��  �          @�Q�@^{@  @�Q�B0�RB@^{?���@�=qBB�A\                                    Bxc�#J  �          @��@g
=@\)@uB(�A�z�@g
=?˅@���B9��A�                                      Bxc�1�  �          @���@s33@$z�@^{B��BG�@s33?�(�@uB'�RAۮ                                    Bxc�@�  �          @�33@u@<(�@P��B�HB��@u@Q�@l(�B  A��                                    Bxc�O<  �          @���@j�H@P��@^{B(�B&33@j�H@)��@|��B$p�BQ�                                    Bxc�]�  T          @�ff@�z�@J=q@VffB�HB�@�z�@%�@s�
B
=A���                                    Bxc�l�  �          @��
@�Q�@H��@C33A�B�@�Q�@'
=@aG�B��A��                                    Bxc�{.  T          @�
=@�\)@w�@�A�B*
=@�\)@\(�@=p�A�Q�B��                                    Bxc���  
�          @�33@�=q@�z�@�A�=qB/{@�=q@o\)@0��A�=qB#��                                    Bxc��z  �          @�{@���@�G�?\AZ{B!{@���@p  @	��A��B�                                    Bxc��   �          @�(�@��H@�ff?��\A8  B)�R@��H@|(�?�A��B"z�                                    Bxc���  
�          @Ǯ@�33@~�R@33A�33B)�@�33@fff@*=qA�B�                                    Bxc��l  T          @�
=@��@p��@
=A�=qB#{@��@U�@;�A��HB�\                                    Bxc��  T          @��
@�ff@mp�@(�A�(�B%�R@�ff@QG�@@  A�\)B�                                    Bxc��  �          @�p�@��\@~�R@A�ffB${@��\@e@-p�A�p�B�\                                    Bxc��^  �          @�(�@��
@���?�(�A�{B��@��
@k�@%A��BQ�                                    Bxc��  �          @��
@�Q�@��\?�A���B(@�Q�@o\)@{A�\)B��                                    Bxc��  �          @θR@��@x��?�A�=qB�
@��@c33@(�A�\)B��                                    Bxc�P  
�          @�ff@���@s33@�
A��HB�H@���@Z�H@(��A�\)B\)                                    Bxc�*�  �          @�{@�ff@e�@�\A�B�\@�ff@Mp�@%A�p�B�
                                    Bxc�9�  �          @�(�@�G�@^�R?�G�A��\B
p�@�G�@I��@33A�
=B 
=                                    Bxc�HB  �          @�ff@��@B�\?�p�A��A��@��@.{@(�A��A�z�                                    Bxc�V�  �          @ȣ�@�(�@<(�?�ffA@  A�33@�(�@,(�?�  A�(�AԸR                                    Bxc�e�  �          @��H@�p�@\(�?�  A6ffB�\@�p�@L��?��A���A��
                                    Bxc�t4  �          @���@���@��?��
A�B ��@���@���?�Q�Ak\)B�R                                    Bxc���  T          @���@��@a�?.{@�33B �R@��@XQ�?�p�A*�RA�ff                                    Bxc���  �          @��@˅@=q>�  @��A��
@˅@?!G�@�33A�\)                                    Bxc��&  �          @�  @˅@
=q>�@y��A�(�@˅@�
?L��@ٙ�A���                                    Bxc���  �          @ڏ\@�z�?Ǯ�B�\���AT  @�z�?���=u>�AUp�                                    Bxc��r  T          @ٙ�@�
=?�33�G����HA�
=@�
=@   ����|(�A��                                    Bxc��  
�          @�(�@�\)@�\���
�
{A���@�\)@
�H�0������A�z�                                    Bxc�ھ  x          @أ�@��@33����6�HA��
@��@\)��  ��A�{                                    Bxc��d  T          @��@��?�Q쿮{�<  Ap(�@��?�녿�����
A��                                    Bxc��
  �          @ҏ\@�  ?�  ����C�AX(�@�  ?��H����
=At                                      Bxc��  "          @��@�(�?:�H����<(�@���@�(�?k���33�(��A	�                                    Bxc�V  T          @��@�=q?
=q���H���\@�(�@�=q?Y��������(�@��R                                    Bxc�#�  
�          @�G�@�\)>�
=�>�R��  @�33@�\)?c�
�8Q���\)A
=                                    Bxc�2�  "          @�=q@��\)�I�����
C�4{@�>�Q��HQ���ff@|��                                    Bxc�AH  �          @��@��׽L���W���C��\@���>��U��@��                                    Bxc�O�  T          @��H@�ff��p��aG��Q�C��@�ff>L���a����@�\                                    Bxc�^�  �          @Å@�(���  �S33��HC���@�(�>�\)�S33�@G�                                    Bxc�m:  �          @��@��\�W
=�Y����\C�� @��\>����X���{@|(�                                    Bxc�{�  �          @��
@��H�#�
�`  �\)C���@��H���
�c�
�\)C�~�                                    Bxc���  
�          @�33@�33��ff�j=q�#ffC��{@�33��ff�q��*  C��R                                    Bxc��,  T          @��H@�G��h���dz��#33C��q@�G���{�j�H�(��C��                                     Bxc���  �          @�Q�@��ͽ����R�\�
=C�Q�@���>�
=�P���@�ff                                    Bxc��x  �          @�\)@��H�(���k����C��@��H��\)�o\)�#
=C���                                    Bxc��  
�          @��@�����q��$�
C���@���ff�y���+Q�C���                                    Bxc���  T          @�33@�33����j=q�"�C�AH@�33�
=q�r�\�*  C�=q                                    Bxc��j  �          @�@�G����H�Z�H�(�C��\@�G��&ff�dz��$Q�C�q�                                    Bxc��  T          @�p�@vff��=q�O\)�  C�0�@vff����W��#�C��3                                    Bxc���  �          @��@\(����
�aG��,=qC�
=@\(��s33�n{�8��C�L�                                    Bxc�\  "          @��\@7���z���=q�Kz�C��{@7��}p���G��Z��C�t{                                    Bxc�  �          @���@x�ÿ�=q�O\)�z�C���@x�ÿ���\���"(�C�~�                                    Bxc�+�  T          @��@y����33�@  �
��C��R@y����33�QG���C��                                    Bxc�:N  
�          @�(�@s33����=p��Q�C���@s33��33�S33�ffC���                                    Bxc�H�  "          @���@p  �#�
�/\)����C��R@p  ��G
=��RC�n                                    Bxc�W�  �          @�ff@g��1G��   ��{C�P�@g����:=q�(�C��)                                    Bxc�f@  �          @�{@P���1G��(����C��3@P����6ff�z�C�"�                                    Bxc�t�  �          @��@1G���ff�C�
�+\)C��H@1G�����S�
�=33C��=                                    Bxc���  T          @�  @Vff�z�H����C��R@Vff�
=�\)��HC�                                      Bxc��2  T          @���@H�ÿ�
=�p���
C�˅@H�ÿ��\�)����C�
=                                    Bxc���  �          @���@=q���1��"Q�C��@=q���H�Fff�9ffC�Q�                                    Bxc��~  �          @�  @{� ���#�
��\C��=@{�z��;��/�HC���                                    Bxc��$  �          @�@33��
=�fff�\��C���@33�W
=�q��n\)C�޸                                    Bxc���  T          @���@33�Ǯ�b�\�OffC��@33�z�H�p  �a{C��H                                    Bxc��p  �          @�(�?�녿��
�g
=�`��C�XR?�녿�
=�vff�y�C�#�                                    Bxc��  �          @���@�\�����n�R�]��C�@ @�\�s33�|(��p��C��                                    Bxc���  
�          @�{@�R���Q��>��C��q@�R�����`���P�C��f                                    Bxc�b  �          @��
@#33��
=�G
=�6
=C�T{@#33��z��U�G�HC�Ǯ                                    Bxc�  �          @�  @z���\���Ip�C��@zῌ���j�H�\G�C�T{                                    Bxc�$�  
�          @��@�����c�
�MC�'�@�����tz��c��C�L�                                    Bxc�3T  �          @��\�333�S33�QG��0ffC�
=�333�/\)�p���TQ�C��                                    Bxc�A�  T          @�=q�
=����c33��HCs+��
=������  �)G�Co�                                    Bxc�P�  �          @�  ��R����dz���\Ct=q��R�~�R�����,\)CpǮ                                    Bxc�_F  �          @�������aG��
��Cu\)���\)���R�,�Cr\                                    Bxc�m�  �          @Å����\)�b�\��\Ct����vff���R�.Q�Cpz�                                    Bxc�|�  T          @�(��\)��\)�a����CsaH�\)�vff���R�-\)Co�                                    Bxc��8  
�          @�p��	����=q�aG��p�Ct�	���|(���ff�,p�Cq\)                                    Bxc���  T          @����G���
=�W��=qCv�H�G����
���H�&��Cs��                                    Bxc���  �          @������z��\����\Cu������������)��Cr�)                                    Bxc��*  �          @��
�Q����R�\���	{Cq�3�Q��vff���
�)G�CnL�                                    Bxc���  �          @��
�\)�����c�
���Co���\)�j�H���R�.{Ck�f                                    Bxc��v  
�          @�=q�=q��=q�`���Q�Cp�\�=q�l������-�Cl�)                                    Bxc��  
�          @�(���R��33�#33��z�Cr�
��R�xQ��N{��Cp
                                    Bxc���  "          @�=q����z������(�Cr������Q��(Q��홚Cpff                                    Bxc� h  �          @��H�*=q�p  �Dz��  Cj�H�*=q�Mp��h���$�\CfG�                                    Bxc�  "          @���� ���2�\�l���2�Cd�� ���
=q��33�L=qC\��                                    Bxc��  "          @����R�*=q��G��E\)Ce����R��(���p��`(�C]aH                                    Bxc�,Z  T          @�p��(��B�\��  �B=qCj.�(���
���_
=Cb}q                                    Bxc�;   
�          @�{�=q�=p��vff�5  Cf���=q��\�����Pz�C_�{                                    Bxc�I�  T          @�z�����R�-p���33Cv���mp��W
=���Ct
=                                    Bxc�XL  
�          @�G����H�L���?\)���Cn}q���H�*�H�^{�9
=Ci�R                                    Bxc�f�  T          @�(��-p��&ff�W
=�'{C_Ǯ�-p��G��o\)�?\)CX�f                                    Bxc�u�  "          @�{�!��.�R�P  �$��Cc.�!��
=q�i���>�C\�=                                    Bxc��>  �          @�z��'��S33�n{�%�
Cg�{�'��)�����R�B  CaY�                                    Bxc���  �          @����:�H�c�
�x��� ��Cf�)�:�H�8Q���p��<��C`�=                                    Bxc���  T          @�  ��<����  �:�
Cg�H�������V��C`�                                    Bxc��0  �          @���"�\��p��:=q���Cp��"�\�x���fff�{Cl�{                                    Bxc���  
�          @���   �����R��ffCq��   ��z��L����Cn�H                                    Bxc��|  
�          @���   �aG��,(��G�Cph��   �B�\�N�R�&z�Cl��                                    Bxc��"  "          @���(��-p��HQ��'�
Cg�(��
=q�b�\�C�HC`�\                                    Bxc���  �          @�(���R�&ff�G��)Q�Ce\)��R�33�`  �D�\C^��                                    Bxc��n  "          @�ff�u�~{�3�
�
{C�4{�u�]p��Z�H�.�RC~z�                                    Bxc�  
�          @�33�E����R�;���Q�C�\)�E�����j=q�#z�C�                                    Bxc��  T          @�
=�@  ��=q�C33��  C���@  ��  �u� z�C�H�                                    Bxc�%`  �          @���(���  �N�R��(�C�����(����������#�HC�\)                                    Bxc�4  
�          @�p��\��(��_\)��RC���\��\)���\�)��C��{                                    Bxc�B�  �          @�z������H�?\)��C�Ф��������u���C��                                    Bxc�QR  T          @����  ���H�1�����C��H��  ����b�\�p�C�S3                                    Bxc�_�  T          @�=q������|���+��C�(����aG���33�R{C��q                                    Bxc�n�  �          @��H������(���Q��#z�C�Z�����y����
=�I��C�<)                                    Bxc�}D  "          @�z�?z�H�ָR�aG��ۮC�"�?z�H�������H���C��
                                    Bxc���  �          @�=q?�  ��33�����p�C���?�  ��(��Z=q����C�S3                                    Bxc���  �          @��@
�H�У��
�H���C�5�@
�H��=q�N{��ffC��R                                    Bxc��6  
�          @�@i����ff�k���(�C��H@i�����H������HC�(�                                    Bxc���  
�          @�G�@p����=q?
=q@�p�C���@p����33�B�\����C�j=                                    Bxc�Ƃ  
�          @��
@x����(�>B�\?�33C�Ф@x����33�
=q��G�C���                                    Bxc��(  "          @\@�ff��=q?!G�@�  C�R@�ff������Ϳp��C���                                    Bxc���  �          @�z�@�=q�x��?Tz�AC���@�=q�~{>#�
?�{C���                                    Bxc��t  �          @�z�@\)���?���AT��C�AH@\)��\)?#�
@�=qC���                                    Bxc�  
�          @�G�@�(��p��?�Q�Ag
=C���@�(��|��?J=q@��\C�"�                                    Bxc��  
�          @�  @hQ����?�  AL  C���@hQ����?�@�z�C�\                                    Bxc�f  
�          @���@Vff��ff?�p�An�\C���@Vff��(�?5@�33C���                                    Bxc�-  
}          @�  @,�����R?��
A$��C�L�@,�����>8Q�?�  C�                                    Bxc�;�  	�          @�ff@5���>�=q@,��C��@5��(�����
=C�R                                    Bxc�JX  
�          @�Q�@'
=��33�
=q��33C���@'
=��{����`  C��=                                    Bxc�X�  �          @��@1���ff=�?�Q�C��@1����ͿB�\���C�,�                                    Bxc�g�  �          @�33@-p���p�>B�\?���C��)@-p����
�0���ٙ�C��{                                    Bxc�vJ  �          @��\@<����Q�>�=q@,(�C�@ @<������z����C�P�                                    Bxc���  
�          @���@-p����H���
�G�C�@-p���  �p����RC�9�                                    Bxc���  �          @�\)@B�\����?�(�AI��C�� @B�\��{>�(�@���C�xR                                    Bxc��<  �          @�Q�@p���mp�@*�HA�=qC���@p�����?��RA�p�C�5�                                    Bxc���  �          @��@p���
=>�  @p�C���@p���{�&ff��{C���                                    Bxc���  �          @�p�?�33��=q�����33C�?�33��(��˅�o�C�N                                    Bxc��.  �          @�  @G�����(����C���@G���p���{�o�
C��\                                    Bxc���  �          @�=q?�����  �\)��=qC���?�����=q��=q�h(�C��{                                    Bxc��z  �          @�\)@
=���?��A!C��H@
=���R=�G�?��\C��\                                    Bxc��   �          @��R@2�\���
@\)A�  C���@2�\��\)?�33A�\)C���                                    Bxc��  �          @��H@J�H����@P��B��C�!H@J�H����@!G�A�  C���                                    Bxc�l  �          @�Q�@N�R�:�H@�z�B.\)C���@N�R�e@e�B
=C���                                    Bxc�&  
�          @��@0  �e�@FffB	�C�@0  ��=q@�Aҏ\C��                                    Bxc�4�  �          @�{@"�\��z�>��?�33C�%@"�\���\�Tz���ffC�AH                                    Bxc�C^  �          @\@�\��{>�ff@�Q�C��R@�\��������C���                                    Bxc�R  �          @�G�?����
=>�
=@|��C�%?����ff�
=��=qC�+�                                    Bxc�`�  T          @���@33����>�  @�C���@33����:�H��Q�C��                                    Bxc�oP  �          @���@   ����>\@c33C��H@   ���׿#�
��
=C��=                                    Bxc�}�  T          @ȣ�?�����Q�?J=q@�Q�C�W
?�����녾�=q�\)C�H�                                    Bxc���  T          @Ǯ?˅��?���A�RC���?˅����<��
>#�
C�e                                    Bxc��B  �          @���?����G�?+�@��C�,�?����녾����h��C�&f                                    Bxc���  �          @�Q�?�\)���?aG�@���C�f?�\)���
��  ���C��)                                    Bxc���  T          @�=q>��R�љ�=��
?+�C�\)>��R�θR��=q��C�aH                                    Bxc��4  �          @�{��{��p�����\)C�����{��Q쿷
=�FffC���                                    Bxc���  "          @�ff?k����<#�
=��
C�o\?k���=q�����
C��                                     Bxc��  �          @�z�@����h��?��A~{C�Z�@����y��?�\)AG�C�j=                                    Bxc��&  T          @�z�@��R��ff@3�
A��C���@��R�p��@+�A��C���                                    Bxc��  �          @�(�@���aG�@L��A�RC��q@����Q�@?\)A�z�C���                                    Bxc�r  �          @��H@�33���@N{A�  C�~�@�33����@<(�A��
C��                                     Bxc�  �          @�=q@���{@L(�A�C�aH@��	��@6ffAי�C���                                    Bxc�-�  T          @ə�@�p���@UB Q�C��=@�p��%@:�HAݙ�C��                                    Bxc�<d  �          @Ǯ@�(�����@p  B�HC�AH@�(��p�@W
=B�C��)                                    Bxc�K
  �          @�Q�@�G����R@tz�B��C�"�@�G��)��@Y��BC��q                                    Bxc�Y�  �          @�33@��
�33@l(�B=qC�� @��
�;�@N{A�C�Ф                                    Bxc�hV  "          @�{@��%@e�B{C���@��L(�@C33A�\C��q                                    Bxc�v�  
�          @��
@��H�H��@P��A�C���@��H�j�H@(��A�C�c�                                    Bxc���  �          @�{@�  �I��@#33A��HC��f@�  �c�
?�Q�A�ffC�G�                                    Bxc��H  �          @�33@����E�@�A���C�Ff@����\��?޸RA~{C���                                    Bxc���  �          @�
=@���_\)@p�A��RC�(�@���w�?��
A�{C��{                                    Bxc���  �          @��@��
�mp�@Q�A��C�� @��
��G�?�AP  C�p�                                    Bxc��:  �          @\@����9��@'�A�  C��@����U�@�\A�G�C�:�                                    Bxc���  �          @\@�p��Vff@33A�=qC�g�@�p��n{?�33A33C��
                                    Bxc�݆  T          @ə�@�Q��~�R@��A�Q�C��H@�Q���33?�\)An�\C�:�                                    Bxc��,  �          @�{@�{�vff@��A�(�C��R@�{��\)?ٙ�A~�HC�\)                                    Bxc���  �          @���@�z��q�@ ��A�
=C���@�z���?��
A�=qC�^�                                    Bxc�	x  �          @��@r�\��  @
=A�p�C��)@r�\��=q?��AQG�C���                                    Bxc�  T          @��R@g
=�u@��A��C���@g
=��?��HAlz�C�h�                                    Bxc�&�  	=          @�p�@_\)�{�@Q�A�Q�C�˅@_\)��Q�?�{A^�HC��=                                    Bxc�5j  
          @���@XQ��qG�@��A�{C��@XQ���(�?\A~=qC���                                    Bxc�D  �          @��\@X���r�\@G�A�z�C���@X����z�?��
A~�RC���                                    Bxc�R�  �          @�@*�H���?=p�@���C���@*�H��
=����C��q                                    Bxc�a\  �          @��@fff��(�?�=qAxz�C���@fff���H?:�H@���C��                                    Bxc�p  �          @�ff@s33�i��@p�A��C�3@s33��  ?��RAqp�C��                                    Bxc�~�  �          @�ff@G���
=?�=qA:�RC�8R@G���33>�=q@;�C��\                                    Bxc��N  �          @�ff@
=��(��#�
���C��H@
=�����aG��Q�C��R                                    Bxc���  �          @�?���G��
=����C�]q?���33��p����C���                                    Bxc���  �          @�?Q���������C��?Q������7
=�=qC���                                    Bxc��@  �          @�  >�\)��=q�!����HC��f>�\)�����U��
C�f                                    Bxc���  �          @�(�?O\)��녿˅���C���?O\)��p��!G���C���                                    Bxc�֌  �          @�G�?�����\)�333��p�C��H?�����Q�����z�C�C�                                    Bxc��2  T          @�=q?��
��\)=�\)?O\)C�T{?��
���ͿTz���HC�y�                                    Bxc���  �          @�{?�����=L��?
=qC�8R?������Q��p�C�b�                                    Bxc�~  �          @�{@���z�\��\)C�@������G��aG�C�t{                                    Bxc�$  �          @�(�@�H����<��
>L��C��@�H��{�Y���(�C�H�                                    Bxc��  �          @�Q�@�����>��@?\)C���@�����\�z���z�C��f                                    Bxc�.p  �          @��@'����?�@�C�~�@'���(����
�e�C�o\                                    Bxc�=  �          @��\@J�H�y��<#�
=uC���@J�H�u��@  �z�C��{                                    Bxc�K�  �          @�z�@Z�H�^{>�G�@�C�H�@Z�H�_\)�u�1G�C�33                                    Bxc�Zb  �          @�{@XQ��w�?333@�ffC��@XQ��|(���\)�=p�C�U�                                    Bxc�i  �          @�33@W
=��G�?uA&�\C��H@W
=����>#�
?�G�C��                                     Bxc�w�  �          @�p�@\)�g
=?:�H@�{C��=@\)�l(�<#�
=�C��H                                    Bxc��T  �          @��@z=q�Z�H?fffA��C�ff@z=q�a�>aG�@Q�C��3                                    Bxc���  T          @���@��
�/\)?�  A�p�C��
@��
�>�R?n{A (�C��q                                    Bxc���  �          @��H@�녿�p�@#33A�G�C�@ @���(�@
=A��
C���                                    Bxc��F  �          @�\)@�\)����@<(�B
=C�5�@�\)���@!G�AۅC�G�                                    Bxc���  �          @Å@�  ��\@X��B  C��f@�  ���@>�RA�C���                                    Bxc�ϒ  �          @��@������@:=qA��
C�S3@����.�R@�HA�33C���                                    Bxc��8  �          @�  @�Q��L��?�33A��\C���@�Q��`��?�p�AP(�C�h�                                    Bxc���  �          @�\)@333�QG�?n{AB=qC�K�@333�X��>��@Y��C��=                                    Bxc���  �          @�33@dz��!G�?���A�33C�c�@dz��/\)?Q�A%C�B�                                    Bxc�
*  �          @��R@8���\)?�A��C�
@8����R?s33A_33C��                                    Bxc��  �          @��Ϳ��\�e��\)���Cz�����\�S33����\)Cy�                                    Bxc�'v  �          @���@A��Dz�?��HA�  C�O\@A��S33?Q�A&ffC�Ff                                    Bxc�6  �          @�G�@$z��Fff?��\A�{C���@$z��R�\?!G�AQ�C��                                    Bxc�D�  �          @��R@�Q��@  @�
A�\)C���@�Q��Vff?�As�C�                                    Bxc�Sh  T          @���@~{�]p�?�33A�p�C�w
@~{�n{?k�A=qC�l�                                    Bxc�b  �          @��@��
�c33?��AeC���@��
�p  ?&ff@���C��{                                    Bxc�p�  �          @��@�(��k�?���A4��C�.@�(��tz�>�{@]p�C���                                    Bxc�Z  �          @�{@�(��r�\?^�RA��C���@�(��x��=��
?W
=C�XR                                    Bxc��   �          @���@�=q�g�?���A�  C�&f@�=q�z�H?�ffA(z�C��                                    Bxc���  �          @�{@xQ��r�\?��HA�C���@xQ�����?fffA�C��                                     Bxc��L  �          @�p�@x���~{?��A(��C�33@x����33>L��@z�C��                                     Bxc���  �          @���@xQ��tz�?Tz�A
�RC�� @xQ��z=q<�>�33C�k�                                    Bxc�Ș  �          @���@a��o\)=#�
>�(�C���@a��j�H�:�H�=qC���                                    Bxc��>  �          @�{@7
=�H���   ��p�C�*=@7
=�#�
�E�(�C�{                                    Bxc���  �          @���@:�H�qG�>�ff@�33C�� @:�H�q녾�{����C���                                    Bxc��  �          @�@,(����>�=q@G�C��f@,(����׿z���G�C��H                                    Bxc�0  �          @��
@/\)���
?�(�A]�C��\@/\)����>�{@tz�C�L�                                    Bxc��  �          @��\@7���(�?&ff@�z�C�k�@7���p��aG��(�C�Ff                                    Bxc� |  �          @�=q?�z��q녿�z���Q�C�4{?�z��Y���
=q��C�8R                                    Bxc�/"  �          @u>����L(��
=q�{C��>����*�H�1��7�HC�'�                                    Bxc�=�  �          @�녾.{�(���C�
�D�C�.�.{��Q��b�\�t�\C���                                    Bxc�Ln  �          @�{��G��:�H�J�H�=(�C����G�����mp��mffC��R                                    Bxc�[  �          @��?u��녿�
=����C��\?u�c�
�/\)�ffC��3                                    Bxc�i�  �          @��R?�Q����ý�G���C��?�Q������{�F=qC�O\                                    Bxc�x`  �          @���?���������H����C�h�?������Ϳ�  ���C��R                                    Bxc��  �          @�p�?���녿����ap�C���?���p��
=��ffC�q�                                    Bxc���  �          @���?�  ��ff�z���ffC���?�  ���R�����\C�\                                    Bxc��R  �          @���?xQ���(��.{��z�C�\)?xQ����
��  ��G�C���                                    Bxc���  T          @�ff?
=���
������C�O\?
=���Ϳ�����RC�t{                                    Bxc���  �          @�(�?��
��  ��(���G�C���?��
��G���p�����C�\                                    Bxc��D  �          @�33?��\��\)�����[�C��=?��\������{�x(�C�H                                    Bxc���  �          @��?��H���;�33��  C���?��H���R�����  C�:�                                    Bxc��  �          @��?Q���\)�����
C���?Q���Q��=q����C��H                                    Bxc��6  �          @��?��\��
=�z��У�C��3?��\��\)�У���p�C�q                                    Bxc�
�  �          @�Q�?����H�   ����C��=?����
�������C��                                    Bxc��  �          @��
?�\)���
�G��Q�C��
?�\)���H������=qC�K�                                    Bxc�((  �          @��
?u��ff�J=q�33C�� ?u����������
C���                                    Bxc�6�  �          @�z�?&ff���R���
�C�C��\?&ff��(�����33C�<)                                    Bxc�Et  �          @\(�@(��0�׿�Q��ɮC�
@(���녿����  C�.                                    Bxc�T  �          @{�@g�?��aG��O�A��H@g�?�=q������A�\)                                    Bxc�b�  �          @u�@j=q=�\)��\)��(�?��@j=q>��R��=q���H@�z�                                    Bxc�qf  �          @���@|(�>aG��=p��)p�@J�H@|(�>\�+��{@�                                    Bxc��  �          @���@~�R=��Ϳ5�"=q?��@~�R>�  �+��p�@g
=                                    Bxc���  
�          @k�@hQ�\)�Ǯ���HC��@hQ�L�;����{C���                                    Bxc��X  �          @S33@QG��aG��aG��xQ�C�)@QG��#�
������
C���                                    Bxc���  �          @b�\@\(��J=q=���?�33C��f@\(��J=q���
���RC��                                     Bxc���  �          @tz�@fff��>��@ƸRC���@fff���R>\)@�C���                                    Bxc��J  �          @�ff@o\)���?�{A�  C�O\@o\)��=q?��Aep�C���                                    Bxc���  �          @^�R@{�5@��B%\)C��@{��Q�@33B�HC�q                                    Bxc��  �          @E@p���p�?�z�BG�C�y�@p�����?��A�33C�Q�                                    Bxc��<  �          @|��@,���7�>�(�@�p�C�� @,���8�þk��S�
C���                                    Bxc��  �          @\)@.�R�2�\�E��3�C�,�@.�R�"�\�������\C�}q                                    Bxc��  �          @�{@8���;��.{�C�O\@8���,�Ϳ�����C�xR                                    Bxc�!.  �          @���@O\)�.�R�:�H���C��=@O\)�   ��=q��Q�C�.                                    Bxc�/�  �          @qG�@p��3�
�#�
�p�C��@p��,�ͿJ=q�C�C�5�                                    Bxc�>z  �          @w
=@/\)�,��>�@ ��C��
@/\)�*=q����{C��                                    Bxc�M   �          @u@���>�R�.{�!�C�b�@���7��W
=�K�C���                                    Bxc�[�  �          @���@H���-p�=L��?!G�C��)@H���)������ffC��\                                    Bxc�jl  �          @g
=@,(��{>�Q�@�
=C�8R@,(��  ����   C��                                    Bxc�y  �          @S33��=q�   �aG���(�C��
��=q��R��
=�C���                                    Bxc���  �          @\(��˅�&ff��G����
Cn�Ϳ˅�\)��=q�ffCj�H                                    Bxc��^  �          @HQ��33�zῈ�����Cj���33� �׿�=q��Q�Cf��                                    Bxc��  �          @0  ��G��&ff�333�nffC�@ ��G������
��z�C��=                                    Bxc���  �          @E����
�6ff���
��G�C��3���
�"�\��z���HC�o\                                    Bxc��P  �          @u�?�ff�J=q���
���C��R?�ff�@  ��ff��G�C��                                     Bxc���  �          @�ff@�
=��=q?�(�A��C�1�@�
=���?�AaG�C�W
                                    Bxc�ߜ  �          @���@�(����?�\)A���C�� @�(��!G�?\A�33C�U�                                    Bxc��B  �          @���@��׾�z�?�ffA�G�C�]q@��׿#�
?�Q�A~�\C�Z�                                    Bxc���  
(          @��\@�녿�{?�Q�AXQ�C�z�@�녿�{?c�
A!�C�R                                    Bxc��  �          @��
@����{?�33A�{C��\@���G�?�G�A�G�C�U�                                    Bxc�4  �          @�Q�@����33?�
=A���C���@���L��?��A��C�W
                                    Bxc�(�  �          @�=q@��H��R?��Ar=qC���@��H�n{?���AQ�C��\                                    Bxc�7�  �          @��H@�=q���?���A��C�S3@�=q�Tz�?�z�Av{C�U�                                    Bxc�F&  �          @�=q@�Q��(�?޸RA�(�C��3@�Q�Tz�?�=qA��\C�S3                                    Bxc�T�  �          @�  @�{���?�G�A�z�C�|)@�{�+�?��A�{C�                                      Bxc�cr  �          @�p�@�(�����?�A�G�C�AH@�(��0��?�ffA�(�C���                                    Bxc�r  �          @��R@��H�(��?�ffA��C�  @��H����?˅A�ffC���                                    Bxc���  �          @�Q�@�\)�u?�ffA�(�C���@�\)�(�?���A��RC���                                    Bxc��d  �          @�{@�Q�    ?��Ao�C���@�Q쾣�
?�ffAhz�C�'�                                    Bxc��
  T          @���@��\�aG�?���A�C��3@��\���?��AmC��                                    Bxc���  �          @���@�33�L��?��
Al(�C��@�33��\?�Q�A[�C�f                                    Bxc��V  �          @��@��׽�\)?���AV�RC���@��׾�Q�?�33AL��C���                                    Bxc���  �          @�Q�@�(�=�\)?�\)AN{?\(�@�(��L��?���AK33C��3                                    Bxc�آ  �          @���@�p�=��
?�z�ATz�?u@�p��L��?�33AQC���                                    Bxc��H  �          @�=q@�p�<��
?�(�A_�
>L��@�p���\)?�Q�AYC�Z�                                    Bxc���  �          @���@��\�\)?���As\)C�33@��\���?�ffAep�C�S3                                    Bxc��  �          @�=q@�=q�L��?˅A�
=C�ٚ@�=q�
=?��RA�(�C��3                                    Bxc�:  �          @�33@��þ�  ?��
A��C��@��ÿ.{?�z�A��C�!H                                    Bxc�!�  �          @�33@�G����
?�G�A���C�(�@�G��@  ?�\)A�p�C�Ǯ                                    Bxc�0�  �          @��@��R��
=?��RA�z�C��R@��R�fff?�=qA�p�C��H                                    Bxc�?,  �          @�  @��H��
=?�(�A���C���@��H�c�
?�A�33C��f                                    Bxc�M�  �          @���@�ff��G�?�Q�A��C�c�@�ff�
=?���A�(�C���                                    Bxc�\x  T          @�G�@�z�<�?��HA�
=>�\)@�z��?�z�A�(�C�l�                                    Bxc�k  �          @���@�
=����@��AхC�h�@�
=�5@A�{C��                                    Bxc�y�  �          @�@��ý#�
?�
=A���C���@��ÿ�?�{A��HC��=                                    Bxc��j  �          @�G�@�Q�5@*�HA�Q�C���@�Q쿬��@��A�ffC���                                    Bxc��  �          @�ff@�
=�
=q@*�HA�RC��R@�
=����@��Aՙ�C��                                    Bxc���  �          @�
=@�{��@1G�A��C���@�{����@#33A�  C���                                    Bxc��\  �          @�Q�@�\)�B�\@,��A�33C�s3@�\)��@�HA��HC���                                    Bxc��  �          @���@�p���=q@E�B33C�AH@�p����@:=qB�\C�aH                                    Bxc�Ѩ  �          @�ff@�����Q�@ ��A�
=C�1�@�����z�?�=qA�=qC���                                    Bxc��N  �          @�33@�=q�
=q?˅A�C���@�=q�\)?}p�A333C���                                    Bxc���  �          @�33@��H�ff?��A�=qC�j=@��H��H?s33A5��C���                                    Bxc���  �          @�@s33�"�\?�  AEC�"�@s33�-p�>��
@z=qC�B�                                    Bxc�@  �          @�\)@j�H��
=?W
=A8(�C��@j�H��>��
@�33C�0�                                    Bxc��  T          @~{@L���   ?�\)A���C�H@L���{?\)A�C���                                    Bxc�)�  �          @Dz�@2�\���>��@0  C���@2�\���
�B�\�a�C��                                    Bxc�82  �          @&ff@{������9p�C�(�@{��ff�(���k\)C��H                                    Bxc�F�  �          @Z=q@N�R�L�Ϳ0���<(�C��@N�R�
=�aG��n�HC���                                    Bxc�U~  �          @X��@H�ÿ�Q�#�
�8Q�C���@H�ÿ��׾�33���RC�3                                    Bxc�d$  �          @Y��@L(���33���
��\)C��@L(���{���
���HC�k�                                    Bxc�r�  �          @{�@hQ쿙���O\)�>�\C��H@hQ�p�׿������C���                                    Bxc��p  �          @mp�@dz�z�>�z�@��
C�^�@dz�#�
>�@�
C��                                    Bxc��  �          @A�@2�\�#�
?��A�ffC�` @2�\��ff?��A��C�c�                                    Bxc���  �          @g�@Tzᾀ  ?��A���C�� @Tz�z�?�
=A�z�C��                                    Bxc��b  �          @|(�@p�׿c�
?+�A��C�S3@p�׿��>��@�G�C�J=                                    Bxc��  �          @�=q@{��s33>�{@��HC�0�@{���G�=��
?�{C���                                    Bxc�ʮ  �          @���@���  <#�
>B�\C�B�@��xQ쾀  �VffC�xR                                    Bxc��T  T          @��@�{�p��>�p�@�33C���@�{��G�=���?��
C��)                                    Bxc���  �          @�(�@��׿aG�>�(�@��\C���@��׿u>8Q�@��C�H�                                    Bxc���  �          @��\@�p��h��?5A  C�(�@�p�����>�ff@��HC�0�                                    Bxc�F  �          @�33@�G����>�(�@��
C��@�G���>�z�@eC�h�                                    Bxc��  �          @�z�@��׿.{?B�\A33C���@��׿\(�?��@�=qC��
                                    Bxc�"�  �          @�p�@�  �c�
?c�
A0  C�g�@�  ����?�R@��
C�'�                                    Bxc�18  �          @��\@�(��z�H?z�HA:�RC�  @�(����H?.{A ��C���                                    Bxc�?�  �          @���@��׿��H?��\AEp�C��H@��׿�Q�?(��@�p�C�'�                                    Bxc�N�  �          @�{@�
=�}p�?W
=A&�\C��f@�
=��
=?
=q@�ffC���                                    Bxc�]*  �          @���@�33��G�?�(�A��
C��@�33��{?���AP��C��R                                    Bxc�k�  �          @��
@��R����?�33A��C�^�@��R���H?xQ�A8��C�z�                                    Bxc�zv  �          @��R@��׿�G�?��A~�\C�G�@��׿�ff?Tz�A"�HC���                                    Bxc��  �          @��\@�{�&ff?G�A�C��@�{�Tz�?�@ڏ\C��{                                    Bxc���  �          @�=q@��þL��?#�
@�(�C�Ф@��þ�p�?\)@�ffC�Ф                                    Bxc��h  �          @�z�@��\�#�
?J=qA�C���@��\�k�?B�\A�HC���                                    Bxc��  �          @���@��R��?Q�A�HC��{@��R��  ?G�A�C���                                    Bxc�ô  �          @�G�@�{��?z�HA4  C��R@�{����?p��A+�C�H�                                    Bxc��Z  �          @�33@��>.{?xQ�A8��@G�@����G�?z�HA:=qC�Z�                                    Bxc��   �          @�{@��=�?Y��A&ff?�ff@����G�?Y��A&�RC�J=                                    Bxc��            @���@|��?
=q?+�A��@�=q@|��>�{?J=qA5G�@�33                                    Bxc��L  �          @p  @g�?=p�?�RA(�A7�
@g�?
=q?L��AE�Aff                                    Bxc��  �          @q�@l��?#�
>�
=@�(�A��@l��?   ?z�A@�ff                                    Bxc��  �          @mp�@h��>��?   @�G�@�(�@h��>��R?(�A��@��                                    Bxc�*>  �          @Z�H@W
=>�z�?�A@�ff@W
==�?!G�A)@�\                                    Bxc�8�  �          @o\)@P�׿�?�  A�Q�C�
=@P�׿Tz�?L��AU��C��H                                    Bxc�G�  �          @��@AG��
�H?У�A�33C�!H@AG��"�\?s33AX��C���                                    Bxc�V0  �          @��@C�
�G�?�z�A�z�C�@C�
�(��?s33ATz�C��)                                    Bxc�d�  T          @�@E��=q?���A�z�C���@E��.{?5A(�C�E                                    Bxc�s|  �          @�(�@A���?�ffA�=qC�7
@A��*�H?Q�A8Q�C�H�                                    Bxc��"  �          @���@=p��?���A�ffC�ٚ@=p��(Q�?(��AffC�,�                                    Bxc���  �          @~�R@;��  ?��HA�{C�@ @;��$z�?B�\A/\)C�`                                     Bxc��n  �          @�33@@  �?��
A�33C���@@  �+�?L��A3\)C�
                                    Bxc��  �          @�G�@A��33?�\)A���C�e@A��%?(��A��C��3                                    Bxc���  �          @��\@@  �?�Q�A�{C��@@  �)��?5A!�C�E                                    Bxc��`  �          @��@A��#33?�Q�A�
=C���@A��;�?c�
A>=qC���                                    Bxc��  �          @���@;���?�A�Q�C��\@;��$z�?xQ�A_�C�c�                                    Bxc��  �          @��\@E����?�A��C�� @E��$z�?5A   C��                                    Bxc��R  �          @�33@J=q�z�?���A���C���@J=q�$z�>��@ӅC�p�                                    Bxc��  T          @�(�@\�Ϳ�?�z�A�=qC��R@\���ff?z�A{C�Z�                                    Bxc��  �          @��
@C33�&ff?(��A�C��f@C33�+����޸RC�XR                                    Bxc�#D  �          @�G�@$z��\(�>#�
@C�` @$z��U��Y���7
=C���                                    Bxc�1�  �          @��@(��X��>�p�@��\C��@(��Vff�!G��33C�                                    Bxc�@�  �          @�  @��j=q��33���C��@��X�ÿ�33����C��                                    Bxc�O6  �          @z=q��G��p�׿5�,  C�!H��G��X�ÿ����ffC��                                    Bxc�]�  �          @j=q��  �>{������ffCs0���  �!녿�33�G�CoB�                                    Bxc�l�  �          @�G���p����
��z����CA����p��녿�33��33C;�\                                    Bxc�{(  �          @���p���
=�����
CE�R��p���p��aG��-��CC�=                                    Bxc�  �          @�33�qG���  �����g33CL�f�qG���{�����Q�CG                                    Bxct  �          @�
=�w����
���
���\CF:��w��0�����޸RC>)                                    Bxc§  �          @��R�W
=��ff����(�CP��W
=��{�{���CFL�                                    Bxcµ�  �          @o\)����(�ÿ��
�噚Cjff��녿��R�\)�(�\CbxR                                    Bxc��f  �          @e������:=q��  ��=qCt���������33�!�CoaH                                    Bxc��  T          @j�H�Tz��S33��ff���HCٚ�Tz��0����R���C}:�                                    Bxc��  �          @tzῺ�H�C33��p�����Ctn���H�{�z��{Coff                                    Bxc��X  �          @��H����8Q��{�хCe�=������(���{C]�                                    Bxc���  �          @�G����h�ÿ��
����Cmn���E��
��33Ci                                      Bxc��  �          @��׿����K����\��(�Cn�=�����)���
�H���Ci�3                                    Bxc�J  �          @����z��I����������Cn�\��z��!G���R�\)Ch�H                                    Bxc�*�  �          @�Q�h���q녿@  �.�HC�=q�h���W���{��ffC~�                                    Bxc�9�  �          @�녿Tz���{����P��C��ÿTz����Ϳ˅��33C�Z�                                    Bxc�H<  �          @�Q쿔z��|(��B�\�&�HC}����z��aG������C{Ǯ                                    Bxc�V�  �          @z=q?&ff�p�׾�\)��p�C��H?&ff�`  ��z���p�C�=q                                    Bxc�e�  �          @c�
>����Y����������C��)>����H�ÿ�{��Q�C��
                                    Bxc�t.  �          @+�?���{?�  A��HC�+�?���>�A�
C�G�                                    BxcÂ�  �          @�\?&ff��Q�?#�
A�Q�C��?&ff�Ǯ>.{@��\C�:�                                    BxcÑz  �          @���>8Q���=q>�(�@��HC�=q>8Q���  �W
=�=p�C�B�                                    Bxcà   �          @���c�
������R�xQ�C�` �c�
��p���
=��  C��
                                    Bxcî�  �          @�z��z��|(����R�y��Cr\)�z��W������z�CnxR                                    Bxcýl  �          @����   �w
=������Cr���   �J=q�0�����Cm�3                                    Bxc��  �          @����p��p�����Csn��Q��Y���$��Cm��                                    Bxc�ڸ  �          @����������&ff���HCvQ����P���s�
�5ffCp
=                                    Bxc��^  �          @�����xQ��N�R��RCs�׿��+������QffCjW
                                    Bxc��  �          @��
��(��@  �g��5�Cl�R��(����H��(��lC]�                                    Bxc��  �          @�ff��Q��*�H�S�
�4Q�Cj���Q��  �~�R�i�CY��                                    Bxc�P  T          @����  �U������HCk��  �%��2�\�Cd�f                                    Bxc�#�  T          @������\�Ϳ�p�����Ck\)����3�
�\)���Ce��                                    Bxc�2�  �          @��H��
�R�\���  Cj���
�p��AG��'ffCb��                                    Bxc�AB  �          @�녿�Q���������Cx\��Q��\���8Q��G�Cs�R                                    Bxc�O�  �          @�
=��  �\)�G��ʏ\Cyc׿�  �I���K��)ffCt}q                                    Bxc�^�  �          @��R���
�|(��33���Cx�=���
�E�L���+z�Cs��                                    Bxc�m4  �          @�����33�b�\���
��  Cu{��33�2�\�3�
�#�HCoz�                                    Bxc�{�  �          @�=q�ff�o\)���Σ�Cp���ff�8���J=q�&33Ci��                                    BxcĊ�  �          @�=q��p��n{�(���CxO\��p��+��l(��IffCq�                                    Bxcę&  �          @��׿�(��\)��\��ffC|�῜(��B�\�\���9Cx#�                                    Bxcħ�  �          @�\)��33�[��@  �C{k���33���|(��d�Cs8R                                    BxcĶr  �          @��ÿ+��\(��5��
C�t{�+��ff�r�\�d�HC}��                                    Bxc��  �          @���<��
�dz�����p�C�.<��
�)���R�\�L��C�>�                                    Bxc�Ӿ  �          @�G���=q�P������\C�� ��=q�ff�Mp��V�\C���                                    Bxc��d  �          @���aG��i���%��
=C�N�aG��'��hQ��X\)C���                                    Bxc��
  �          @�녿�����ff����  Cy�������N�R�aG��1�RCt)                                    Bxc���  �          @��R���r�\�{���
CnxR���2�\�c�
�1�HCf#�                                    Bxc�V  �          @�����H�]p��Y���"  Cp�����H�
�H���H�`=qCc�                                    Bxc��  �          @�33����C33�o\)�<�Cq��녿�z������{
=Cak�                                    Bxc�+�  �          @��O\)�<���`���C��C~���O\)���������Cs��                                    Bxc�:H  �          @�  ���1��333�3z�C�J=�����H�c�
�~Q�C~�H                                    Bxc�H�  T          @y���J=q�Q��>�R�GG�C{�׿J=q���
�e��Cn.                                    Bxc�W�  �          @�{�fff����>\@�Q�C�Z�fff���Ϳ����a�C�33                                    Bxc�f:  �          @��ÿ:�H���R=�\)?G�C����:�H��
=���R��C�]q                                    Bxc�t�  �          @�Q�>����
=������RC��\>����G��
=���
C��                                    BxcŃ�  �          @������ff��Q�����C�=q�������^{�"�RC��                                    BxcŒ,  �          @�\)�������R�  ��p�C�������N{�aG��8�C{�{                                    BxcŠ�  �          @�G���H�k��W
=��Cl�f��H�ff�����Q�C`(�                                    Bxcůx  �          @��R�33�u�����ffCq��33�8���Vff�-�Cj�                                    Bxcž  �          @�p���z���ff���\�g\)C}\)��z���Q��0���
=Cz��                                    Bxc���  �          @��>�������c�
�=qC��f>�����Q��(Q�����C�                                      Bxc��j  �          @�����������
=q��=qC�o\������Q��n{�+\)C�K�                                    Bxc��  �          @��
=�\)��  �R�\�C�u�=�\)�HQ�����_(�C��=                                    Bxc���  �          @�p���ff�u�i���*G�C~�3��ff�Q���\)�t��Cv=q                                    Bxc�\  �          @��333�����H�˙�Cl�\�333�K��mp��$�RCd��                                    Bxc�  �          @��
�ff�e��e���Cl���ff�
=q���\�\\)C^�{                                    Bxc�$�  �          @�z�У��N{��Q��F��Cs#׿У׿˅��33��C`J=                                    Bxc�3N  �          @�G���z��@����p��i�
C�AH��z῔z�����{C��                                    Bxc�A�  
�          @�=q��Q쿌����=qu�CUͿ�Q�?E���(��)Cz�                                    Bxc�P�  �          @�
=��    ��Q�ǮC3�
��?����{W
C aH                                    Bxc�_@  
�          @�zῥ���
=���\�z(�Chff����B�\������C<G�                                    Bxc�m�  �          @�33�c�
��(���\)�
Cn�׿c�
=u��\)�=C/��                                    Bxc�|�  �          @�G��z�#�
��ff¥��C5W
�z�?�����p�  B���                                    BxcƋ2  �          @���   =�Q���{§��C*+��   ?�Q����
#�B�                                      Bxcƙ�  �          @��\��33<��
����«8RC1^���33?�����  �B��                                    Bxcƨ~  �          @�=q��G�<��
����¨��C1)��G�?�=q��\)�RB�\)                                    BxcƷ$  �          @�  =#�
?.{����¢�B�
==#�
@	���y���tffB���                                    Bxc���  �          @�z�>�z�?Y����G��B�>�z�@��~{�l{B�8R                                    Bxc��p  �          @�z�>���?:�H���� 
=B���>���@����Q��q�B�z�                                    Bxc��  �          @�=q?u?�\����~p�Bu��?u@Dz��L���333B�33                                    Bxc��  �          @���?J=q?�=q�����)B}��?J=q@;��U�?=qB���                                    Bxc� b  �          @�G�?8Q콸Q���\) �\C��3?8Q�?���\).Bz��                                    Bxc�  T          @�
=>�(�>�����z�¥  B
=>�(�?�G��qG���B�p�                                    Bxc��  �          @�p�>.{?O\)��=qB���>.{@��a��h�B��)                                    Bxc�,T  �          @�  ?L��?�{�n�R�v(�B�u�?L��@B�\�5�(Q�B���                                    Bxc�:�  �          @��R?�33?���j�H�u��B\(�?�33@4z��7
=�,�
B���                                    Bxc�I�  �          @��H?��?5�z�H8RBP33?��@�\�[��i33B��q                                    Bxc�XF  �          @x��>�
==����vff§=qA_�
>�
=?����dz�\B��                                    Bxc�f�  �          @��H>�Q�>�����G�¤�HB=��>�Q�?�ff�i���}33B�.                                    Bxc�u�  �          @����#�
?8Q��|��L�B��H�#�
@z��\(��k�
B�\                                    BxcǄ8  �          @��׽u?�{�u��B�Q�u@���L���T�RB��                                    Bxcǒ�  �          @�{?��
����xQ��zffC��\?��
>�����H�f@�p�                                    Bxcǡ�  �          @��?(��fff�vff� C�3?(�?��z�H�qB#�                                    Bxcǰ*  �          @xQ�?�ff��=q�Tz��r��C�#�?�ff�u�e�  C��{                                    BxcǾ�  �          @xQ�?h��?��`���B �\?h��?���Dz��c�\B{�                                    Bxc��v  �          @��?����W
=�l���qC���?���?����a��z  BG�                                    Bxc��  T          @��?��R�(��3�
�233C��{?��R���\�`  �s�C��                                    Bxc���  �          @��?��AG���R���C���?���
=�Y���ap�C��                                     Bxc��h  �          @�(�?z��   �aG��m��C��?z�����Q�p�C��                                    Bxc�  �          @�{?�33��G��b�\�C�*=?�33=����p  �)@��                                    Bxc��  �          @�{@Q쿗
=�b�\�TG�C��@Q�>.{�n�R�e\)@���                                    Bxc�%Z  �          @�33@5�����L(��@�C��
@5?Y���Fff�8�HA��H                                    Bxc�4   �          @�{@�R>k��p���l�@�33@�R?����Z�H�M�
B�                                    Bxc�B�  �          @��R?��H>u��{#�A�?��H?޸R�tz��m{BH                                      Bxc�QL  �          @��
?ٙ�>�\)�vff�RA�H?ٙ�?�33�_\)�_  B0�                                    Bxc�_�  �          @��?�p�=����s�
�z
=@2�\?�p�?�p��`���[��B                                    Bxc�n�  �          @�p�?��>�{�}p��{A"�\?��?�G��dz��X�
B,\)                                    Bxc�}>  �          @�{?5>�(����HaHA��H?5?�p��xQ��v(�B���                                    Bxcȋ�  �          @�  ?(�?�\��p��\B�?(�@�z�H�s��B�\)                                    BxcȚ�  �          @�ff?�\)�\)�����C�  ?�\)?\��\)u�B?�H                                    Bxcȩ0  �          @���?��׾�(�����C��R?���?�����
==qB:p�                                    Bxcȷ�  �          @�=q?h�þ����)C�{?h��?�
=�����RBQ��                                    Bxc��|  �          @��@�\��G��w
=�x=qC���@�\?����hQ��`�B�\                                    Bxc��"  �          @�{@#�
�8Q��dz��Y=qC���@#�
?�33�X���I�A�33                                    Bxc���  �          @�p�@/\)���Z=q�KG�C�{@/\)?Q��Vff�E�A��                                    Bxc��n  �          @�(�@0      �Y���K�H=�Q�@0  ?�G��I���8��A���                                    Bxc�  �          @��@ff�����k��p
=C���@ff?���a��`�A�
=                                    Bxc��  �          @�@=p�>���QG��>G�@�z�@=p�?˅�8Q��"z�A�z�                                    Bxc�`  �          @�p�@G�>����E�2  @�33@G�?\�.{�Q�A�(�                                    Bxc�-  �          @�(�@7
=>8Q��i���Oz�@e@7
=?��
�Tz��6A�R                                    Bxc�;�  �          @��@:=q�#�
�h���M��C�� @:=q?����X���:�HA��
                                    Bxc�JR  �          @��H@C�
�#�
�Z�H�@�C���@C�
?���N�R�2�A��H                                    Bxc�X�  �          @�Q�@Mp�����_\)�==qC���@Mp�?����U�2ffA���                                    Bxc�g�  �          @�  @P�׾��
�[��9(�C�'�@P��?z�H�S33�0\)A�ff                                    Bxc�vD  �          @�{@S33<��Tz��4�R>��@S33?��\�Dz��#A��R                                    BxcɄ�  �          @�\)@a�>u�Dz��#��@z=q@a�?����0���
=A��R                                    Bxcɓ�  �          @��R@z��\��G��n=qC���@z�?�ff�|(��dffA¸R                                    Bxcɢ6  �          @���@
=�(����(��s(�C�.@
=?������l\)A���                                    Bxcɰ�  �          @��H@�R�W
=��{�qffC���@�R?O\)��ff�r
=A��R                                    Bxcɿ�  T          @���@3�
�����n{�R��C��\@3�
?����c�
�F=qA��H                                    Bxc��(  �          @�G�@i��>�G��Dz����@�p�@i��?����+���
A��H                                    Bxc���  �          @�@qG�>����I���(�@\@qG�?����1G���\A���                                    Bxc��t  T          @�G�@_\)?��L���)
=A�H@_\)?ٙ��0�����Aϙ�                                    Bxc��  �          @��@   >B�\�u��c33@���@   ?У��^{�E�B��                                    Bxc��  �          @��H@Dz�����U�<��C�Ff@Dz�?h���O\)�5�A��                                    Bxc�f  �          @�\)@HQ쿊=q�(Q���C��H@HQ�<#�
�5�)  >L��                                    Bxc�&  �          @��H@n�R��z��*�H�{C��@n�R?B�\�$z��ffA7�
                                    Bxc�4�  �          @�{@E��k��J�H�6�HC���@E�?�  �@���+A�{                                    Bxc�CX  �          @�z�?����(��~�R�{C��f?��>�������ffA��H                                    Bxc�Q�  �          @��
����`���^�HCp���z�����B�CNB�                                    Bxc�`�  �          @��
��
�%�)����Cd@ ��
��\)�\(��PffCR�
                                    Bxc�oJ  �          @������4z��N{�*Q�Ci�{����{����j{CU#�                                    Bxc�}�  �          @��H��
�6ff�;��p�Cf���
��  �s�
�XffCT޸                                    Bxcʌ�  �          @�  �Z�H�A��   ��
=C]���Z�H�G��AG���
CR�{                                    Bxcʛ<  �          @�ff�E�<(��p���\)C_�=�E��G��Y���.�CQ�)                                    Bxcʩ�  T          @�\)�\)�5��A�� G�Cg��\)��
=�x���^�CT�
                                    Bxcʸ�  �          @�z���
�,���HQ��*�\Ch���
���\�{��i�CS�3                                    Bxc��.  �          @��׿��H����E�9Q�Cjz���H��G��q��y
=CRs3                                    Bxc���  �          @�33��G��  �C33�A��Cl+���G��aG��l(�L�CRO\                                    Bxc��z  �          @��R�\�33�S�
�Qp�Cin�\�
=�vffCIQ�                                    Bxc��   �          @�  �s33��{�mp��rffCr�׿s33�����z�.CCL�                                    Bxc��  �          @������!녿������Ca���Ϳ˅�.�R�+��CT��                                    Bxc�l  �          @���[�����Q���Q�CTff�[�����=q���CIW
                                    Bxc�  �          @�z��#�
�(Q��=q�ӮCa�q�#�
�ٙ��.{�%�RCU�\                                    Bxc�-�  �          @��\��\)�:=q�Q���G�Cm8R��\)��=q�Fff�GQ�C`ff                                    Bxc�<^  �          @|(����\)��33��Cbp�������.�R�0  CU�                                    Bxc�K  �          @���<���#33��
=�ҸRC\�\�<�Ϳ�=q�1����CP&f                                    Bxc�Y�  �          @�G��'��'
=������HC`�R�'���ff�@  �2�CR��                                    Bxc�hP  T          @����#33�,(��	�����HCb���#33��\)�B�\�4�CTh�                                    Bxc�v�  �          @��H�+��+��&ff�	��C`��+���
=�\���B��CO��                                    Bxc˅�  �          @�  �(��5�8Q��33CeE�(����H�qG��S�RCR�H                                    Bxc˔B  �          @����7��Dz��2�\�Q�Cc  �7����H�q��BG�CR�{                                    Bxcˢ�  �          @����@���QG�������Cch��@���33�^�R�.��CVY�                                    Bxc˱�  �          @���1G��'��!��z�C_^��1G�����W��=z�CN��                                    Bxc��4  �          @��
�2�\�/\)�ff��  C`���2�\��=q�P  �5�RCQ�                                     Bxc���  �          @���<(��fff��
=��33Cf���<(��.�R�0  �
  C^�H                                    Bxc�݀  �          @�  �G
=�_\)�(����\)CdQ��G
=�<(��   ��z�C_^�                                    Bxc��&  �          @����K��\(��s33�733Cc=q�K��1G��  ���\C]�                                    Bxc���  �          @�  �Vff�G
=���H�lz�C^���Vff���ff���HCW8R                                    Bxc�	r  �          @�\)�e�&ff�������RCW�3�e�޸R� ��� ��CM�)                                    Bxc�  �          @��\�`  ��R��{��33CWL��`  ��(��  �CN(�                                    Bxc�&�  T          @�{�8���P�׿.{��Cds3�8���-p���Q���z�C_&f                                    Bxc�5d  �          @�G��9���=q�������\C[�3�9����\)�33�
�HCQ!H                                    Bxc�D
  �          @r�\�B�\��\)���
��RCL33�B�\������C=�f                                    Bxc�R�  �          @�z��J=q�
=q�#33��C=�J=q?
=q�#33�
=C*c�                                    Bxc�aV  �          @�(��Z�H���
�.�R�=qC4aH�Z�H?�=q�   ��\C"ff                                    Bxc�o�  �          @��R�g
==�\)�%��
=C2���g
=?�{������C"��                                    Bxc�~�  �          @�(��q�?���2�\�ffC ٚ�q�@�� ������CQ�                                    Bxc̍H  �          @����Y���dz��G���p�Cbff�Y���#33�Dz��G�CX��                                    Bxc̛�  �          @����Vff�q녿����  Cdp��Vff�4z��<����HC\�                                    Bxc̪�  �          @�\)�U��l(��������Cc��U��.�R�<(��	33C[W
                                    Bxc̹:  �          @���Dz��Z�H�����Cd!H�Dz��"�\�,���\)C[�f                                    Bxc���  �          @���.�R�_\)����=qCh�.�R�&ff�.�R�p�C_��                                    Bxc�ֆ  �          @���-p��Z�H��\)��33Cg��-p��p��8Q��
=C^@                                     Bxc��,  T          @���'��Z�H��p���  Ch� �'�� ���0����
C_�R                                    Bxc���  �          @�p��(��tz��G����
Cp)�(��7
=�<���33Chz�                                    Bxc�x  �          @�\)�ff�u���H���Cn���ff�9���:=q���Cf�q                                    Bxc�  �          @����c33�����{33Cl:����/\)�#33�Ce�                                    Bxc��  �          @�Q��%��_\)��z��n�HCi�=�%��-p��   ��CbW
                                    Bxc�.j  �          @���@���`  ��{�W�
CeB��@���.�R������HC^(�                                    Bxc�=  �          @��R�@���J�H�aG��6{Cb�\�@���!��
=��ffC\
=                                    Bxc�K�  �          @�ff�B�\�J�H�=p��z�Cb:��B�\�%���p��ӮC\\)                                    Bxc�Z\  �          @�33�{���ff>���@���CL�H�{����
��ff��\)CLL�                                    Bxc�i  �          @����������G����RCL�=��녿��H��G��h(�CH��                                    Bxc�w�  �          @��H��ff��33>L��@�\CJ{��ff���ÿ(���(�CI&f                                    Bxc͆N  �          @�����\)����?
=@�\CK����\)��
=�k��5�CL�                                    Bxc͔�  �          @��{�����?E�A ��CM.�{��   ��\)�n{CN�R                                    Bxcͣ�  �          @��\�s33��Q�?(�Ap�CO�s33�G�����[�CO�                                    BxcͲ@  �          @�z���=q���>��R@r�\CPff��=q�Q�#�
��{CO�
                                    Bxc���  �          @�(���G��.�R=#�
>�G�CV���G��\)��{�Q�CS��                                    Bxc�ό  �          @�{�w��%�>��@�G�CU���w��"�\�&ff� z�CUL�                                    Bxc��2  �          @�G��}p����=#�
?   CQ��}p�� �׿c�
�4��CN�3                                    Bxc���  �          @z=q�`�׿�ff��  �eCO8R�`�׿��
�z�H�f{CK�)                                    Bxc��~  �          @{��l(���G�=�G�?�ffCF�
�l(���Q��(���(�CE�{                                    Bxc�
$  �          @s33�W���{>�
=@�p�CM���W���녾�������CM�3                                    Bxc��  �          @����_\)��\�(��{CN���_\)���Ϳ�ff���CI0�                                    Bxc�'p  �          @w����2�\�u�h��Ce����Q��33���C]B�                                    Bxc�6  �          @z�H�ff�(�ÿ���
=Cgh��ff��=q�6ff�==qCX�f                                    Bxc�D�  �          @QG��G��\)�  �-�
Cz�G���=q�>�R�)Cj
                                    Bxc�Sb  �          @e��z�H�2�\�G��	��Cz�ÿz�H���@���e33Co�3                                    Bxc�b  �          @fff�@  �;�����RC���@  ���>�R�aG�Cw                                    Bxc�p�  �          @\�ͿxQ��.�R��\)�=qCz�ͿxQ��
=�6ff�_{Cp&f                                    Bxc�T  �          @e�p���*=q�(���\Cz��p�׿�(��G
=�r��Cm�\                                    Bxc΍�  �          @p�׽L���*=q�'��2(�C�s3�L�Ϳ�ff�`  aHC�޸                                    BxcΜ�  �          @��\�E��hQ������
=C�׿E��Q��^�R�Y�C|\                                    BxcΫF  �          @����Z=q����{C�"�����
�g
=�n�C��\                                    Bxcι�  �          @��\<��
�R�\�Q��=qC�#�<��
��(��c33�t=qC�<)                                    Bxc�Ȓ  �          @��ü#�
�C�
�%�� �RC���#�
��z��h���C��                                    Bxc��8  �          @h��<#�
�,(����� 33C�&f<#�
��(��L(��RC�Ff                                    Bxc���  �          @|(��(��p�׿aG��P��C�lͿ(��AG��=q�  C�T{                                    Bxc��  T          @r�\�\)�fff�}p��rffC����\)�5���� ��C�n                                    Bxc�*  T          @~�R��Q��u�n{�W�
C�T{��Q��E��\)�
=C��=                                    Bxc��  �          @��;L�������J=q�/�C����L���Tz�����z�C�@                                     Bxc� v  T          @�=q��\)���R�p���Ip�C�� ��\)�Y���*=q�33C�aH                                    Bxc�/  �          @���Q���G���=q�p(�C�ZὸQ��K��.{�!��C�/\                                    Bxc�=�  �          @��
�.{�~{�����xz�C��=�.{�G
=�-p��${C�t{                                    Bxc�Lh  �          @��R������^�R�>{C�����U�#33��RC�Ǯ                                    Bxc�[            @�33��\)���׿8Q��z�C�&f��\)�b�\�\)�33C��                                    Bxc�i�  �          @�
=�O\)�����Y���0��C��=�O\)�`���(Q��z�C�~�                                    Bxc�xZ  �          @�ff�   ���\�Tz��,Q�C��q�   �c33�'���HC�f                                    Bxcχ   �          @��H������녿Y���6=qC~h������R�\�!G��C{�                                    Bxcϕ�  �          @��
���
���׿p���IC|W
���
�N{�&ff�p�CxG�                                    BxcϤL  �          @��������\(��>ffC�N���Q��"�\�z�C�q�                                    Bxcϲ�  �          @�z��ff���׿aG��D��C�˅��ff�O\)�"�\�p�C�f                                    Bxc���  �          @�
=��{��Q�!G��
�\C~����{�U��
�p�C{��                                    Bxc��>  T          @�p�������R�=p���C�f����^{�\)�
{C}�                                    Bxc���  �          @��
�\(���G���\)��ffC���\(��B�\�?\)�-��C~+�                                    Bxc��  �          @w
=���
�Tz������{C|�=���
��\�8���D33Cu��                                    Bxc��0  �          @�
=��z���녿�Q��v�RCz�f��z��H���5�G�Cu�
                                    Bxc�
�  �          @�  ��33������  �MG�Cwٚ��33�Mp��*�H��HCr�=                                    Bxc�|  �          @�����
��zῪ=q��  Cy�R���
�H���@���"��Ct
=                                    Bxc�("  �          @����˅���ͿO\)�%p�Cy\�˅�XQ��#33�	=qCt�\                                    Bxc�6�  �          @�G����
��
=�.{�
{Cz����
�_\)�p��ffCvc�                                    Bxc�En  T          @����=q��G��h���3�
Cy�Ϳ�=q�]p��,����\Cuu�                                    Bxc�T  
�          @�
=��������.{��\C@ �����H���R�gp�CvaH                                    Bxc�b�  �          @��\(��z�H�3�
��C�Ф�\(���\��  �p\)CyaH                                    Bxc�q`  
�          @��׿�z��|(���Q���\)C}�)��z��AG��3�
�#�
Cx�R                                    BxcЀ  �          @���Q��p�׿�{���RC|k���Q��1G��8Q��.��Cv                                    BxcЎ�  �          @��\���R�vff���
��{C|.���R�1��Dz��5�Cv                                      BxcНR  �          @�p��aG��|(��ٙ���33C��R�aG��1��QG��@��C|}q                                    BxcЫ�  �          @�(��Y���mp��Q���=qC��׿Y���Q��c33�Z
=CzL�                                    Bxcк�  �          @�p���ff�vff����\)C~��ff�)���U��E�\Cxk�                                    Bxc��D  �          @����
�}p�����[33Cx�����
�Fff�+��\)Cs�R                                    Bxc���  �          @��Ϳ��������s33�H(�C{�H�����Mp��(�����CwE                                    Bxc��  �          @�{������(���G���p�Cy�f�����n�R���Cv�                                     Bxc��6  �          @��
��Q���=q��  ��z�C~�
��Q��Fff�_\)�9��Cx��                                    Bxc��  �          @�����������\��
=Cy(�����.�R�hQ��D�Cp��                                    Bxc��  �          @���  ��ff�����p�Cz\)��  �K��B�\�#��Ct�                                     Bxc�!(  �          @�{��{��ff������CyͿ�{�K��B�\�!��Cs(�                                    Bxc�/�  T          @���Q���=q���R��(�C��f�Q��L���P���2z�C��                                    Bxc�>t  �          @��k���Q��=q���C�uþk��@  �b�\�F��C�Ф                                    Bxc�M  �          @�33�#�
��G�����ޣ�C���#�
�5��|(��Y(�C�ff                                    Bxc�[�  �          @�\)�(�����
�G��مC��\�(���9���~�R�U�C��                                    Bxc�jf  �          @�  �.{����������C����.{�0  ���
�]C��                                    Bxc�y  �          @����
=�x���7
=��C��H��
=��������y�
C���                                    Bxcч�  �          @�Q����G
=�Mp��6��C�H���녿��������C}:�                                    BxcіX  �          @�ff�����O\)�\(��_�Cy�Ῑ��� ���{�Q�Ct�                                    BxcѤ�  �          @��R��{����?�\)AQG�Czc׿�{�������
�A�CzxR                                    Bxcѳ�  �          @������H���R?   @��RCz\���H����z����\Cx�                                    Bxc��J  �          @�=q������?�  A�p�C�k�������ͿJ=q���C��=                                    Bxc���  �          @��
�(����G�?��
A���C�q�(�����׿J=q��C��R                                    Bxc�ߖ  �          @�ff��G����H?�\)A��C��f��G���  �s33�ffC���                                    Bxc��<  T          @�(������  >���@EC�����������p���\)C�*=                                    Bxc���  �          @�33�޸R���þ�����Cz�׿޸R����(����ffCw��                                    Bxc��  �          @�{��\)���.{��C|�H��\)�����\)��(�CzaH                                    Bxc�.  �          @��\�����
��(���(�C~� �����
�,(����HC|                                    Bxc�(�  �          @�(���ff����333��ffC}���ff����;���HCy�f                                    Bxc�7z  �          @���{���Ϳ�Q��K\)C��Ϳ�{����X���ffC~��                                    Bxc�F   �          @�p��\��  ��p��|��C}��\�r�\�e��$�Cx�                                    Bxc�T�  B          @�녿Y����33�G���Q�C�C׿Y���h������>��C�k�                                    Bxc�cl  �          @�������  �У����CǮ�����`  �g
=�1
=C{                                      Bxc�r  
�          @�
=������Q������
C}
�����c33�b�\�*�
CwǮ                                    BxcҀ�  �          @�  ���R��\)�+���  C~\���R����>{��C{{                                    Bxcҏ^  �          @��H���R��
=���w33C����R�q��aG��%C{�=                                    BxcҞ  "          @�  �\���\����m�C|xR�\�l(��X���!Q�Cw��                                    BxcҬ�  �          @��R���
���׿�ff���HCxs3���
�L���j=q�3�RCp��                                    BxcһP  �          @�����
��\)��\)�r{CyT{���
�e�W�� (�Cs��                                    Bxc���  �          @�=q��33���R��G���p�Cz����33�X���mp��2�Ct�                                    Bxc�؜  
�          @��
�Tz������0������C��
�Tz��1G������e�C}p�                                    Bxc��B  �          @�p���33��  ��\)��
=Cy:��33�mp��n{�'(�Cr�                                    Bxc���  "          @����R����<����ffCw33���R�8Q���G��WG�CkE                                    Bxc��  "          @�녿У�����E�� �HCzuÿУ��,(����H�c�
Cn�q                                    Bxc�4  �          @�녿����(��N{��C}�)����)����\)�l��Cs(�                                    Bxc�!�  �          @��ÿ������c33���C� ����33���~\)Cs��                                    Bxc�0�  T          @�=q�G������Vff��C�@ �G��&ff����x
=C}W
                                    Bxc�?&  �          @��H���R���\�@������C�)���R�:�H��z��d33Cv�R                                    Bxc�M�  "          @�(���\�qG������3��Ct�H��\������p�=qC[^�                                    Bxc�\r  !          @���33�E���ff�Oz�Cq�Ϳ�33�333���{CJ޸                                    Bxc�k  
�          @�(��   �/\)��
=�S(�Ciٚ�   �\��\)ffC>޸                                    Bxc�y�  
�          @��\��{�aG���ff�=\)Cus3��{������33�)CX��                                    Bxcӈd  �          @����ff��z��Q����C}}q��ff������tz�Cqp�                                    Bxcӗ
  �          @��������R�C33���C~ٚ���������r��Ct                                      Bxcӥ�  
�          @��R�z�H�����7����
C����z�H�,����(��h�\Cz�                                    BxcӴV  �          @��׿^�R���H�<(�� ��C��H�^�R�.{��
=�kQ�C|@                                     Bxc���  T          @��^�R��z��C�
�	p�C�` �^�R�\)����s��Cz��                                    Bxc�Ѣ  �          @��H�=p��h���^{�,33C�7
�=p��У���Q��Cuu�                                    Bxc��H  
�          @�녿=p��g��[��+��C�.�=p��У����RffCuff                                    Bxc���  
�          @��\�p���o\)�S�
�"z�C��p�׿�����\Crff                                    Bxc���  
�          @�{�n{�s33�Y���#�
C�!H�n{�������HCr�                                     Bxc�:  "          @�p��L���$z��x���]ffC|�)�L�;���z�.CR��                                    Bxc��  
�          @���p���W��Z�H�1Q�C~c׿p�׿�z����Hu�Cl=q                                    Bxc�)�  
�          @���G��1��{��T
=Cy�῁G��&ff�����CT��                                    Bxc�8,  
�          @�=q�����_\)�\(��+�HCz�R������  ����CgO\                                    Bxc�F�  �          @����G��b�\�]p��*�\CzT{��G����
��ff.Cf�                                     Bxc�Ux  �          @�zῚ�H�Fff�z=q�F��Cx�R���H�n{����p�CY��                                    Bxc�d  
�          @�33��G��B�\�x���F�HCws3��G��c�
�����CWL�                                    Bxc�r�  �          @�����:�H�n�R�G�Cx.���Y������CX&f                                    Bxcԁj  
�          @�  ��  �.�R�o\)�P�Cy𤿀  �.{���\z�CV^�                                    BxcԐ  
�          @�Q쿕�>�R�B�\�.33Cx�{����p����\�HCbaH                                    BxcԞ�  T          @�G���G��l(��!���p�Ct}q��G��ff�|���]{Ce�R                                    Bxcԭ\  �          @���
�H�g
=�+���RCo��
�H���������X�
C^�                                    BxcԼ  "          @�G����a��*=q����CkaH������\)�Q�CY�                                     Bxc�ʨ  "          @����H�S33�5��
��Ci�q��H��{��G��Y�CU��                                    Bxc��N  "          @���z��C33�E��p�Ck�f�zῢ�\�����n��CS��                                    Bxc���  �          @�33��\�P  �HQ�� ��Cqff��\�������x�CZ�q                                    Bxc���  
�          @�����p��C33�_\)�:��Cw�R��p�������Q�#�C\��                                    Bxc�@  
�          @��H�ٙ��XQ��AG��z�Cs=q�ٙ���=q��  �uQ�C^ٚ                                    Bxc��  �          @��\��33�S33�=p���\Cp
=��33�������n33CZ�                                    Bxc�"�  �          @����E��E�%p�Cq}q�����
��p��|��CY�                                    Bxc�12  �          @��R�У��C33�Mp��+�Cq�H�У׿�����Q�=qCXT{                                    Bxc�?�  T          @��Ϳ����C�
�QG��4ffCy����׿�
=���\��Cb{                                    Bxc�N~  �          @�zῠ  �HQ��J�H�,��Cx5ÿ�  �������\)Ca��                                    Bxc�]$  "          @�{���e�   �33Cu���� ���x���`�CfB�                                    Bxc�k�  �          @��\��z��\���'��z�CwǮ��z�����{��np�ChO\                                    Bxc�zp  �          @�=q��G��N{�>�R�"�HCx�f��G����H��z�#�Ce�                                    BxcՉ  �          @�  �����hQ��33��ffCy�Ῠ���Q��o\)�`z�Cn(�                                    Bxc՗�  �          @��Ϳ����_\)�\)���Cwn�������g��]z�Cj�                                     Bxcզb  "          @�zῑ��Z=q�   �{C{xR��녿����s�
�q33Cn33                                    Bxcյ  �          @��R��
=�j=q��R��HC����
=�z��y���vp�C�C�                                    Bxc�î  �          @�ff��z��O\)�/\)�G�Cz\)��z�����|(��~�Ci�R                                    Bxc��T  T          @�녿@  ���Ϳ&ff���C�1�@  �e��(���G�C�                                      Bxc���  �          @�G��Tz���ff��Q���ffC�o\�Tz��B�\�O\)�7\)C~�                                    Bxc��  �          @��\����}p����Ǚ�C}�q����&ff�c33�M��Cv^�                                    Bxc��F  T          @����  �b�\�'��	G�Cw\��  ����~�R�k
=Cg��                                    Bxc��  �          @��Ϳ���c�
�*�H�\)C|B���녿�������up�Cn�H                                    Bxc��  �          @�
=�aG��l���1G��\)C�Y��aG���(���{�{=qCu�q                                    Bxc�*8  �          @�\)�#�
��Q��=q���C�|)�#�
�����Q��hC(�                                    Bxc�8�  
�          @�Q�\)�{��'
=�=qC��R�\)�\)�����s��C�f                                    Bxc�G�  �          @�p�����Z=q�J�H�+��C��)����\������C�*=                                    Bxc�V*  
�          @��
���H�H���C�
�0  C�����H�������Cy�q                                    Bxc�d�  �          @�
=�   ��\)��\)��G�C��Ϳ   �S�
�Tz��3{C��                                    Bxc�sv  �          @�=q�������
�=q��C��
�����   ����h��C�AH                                    Bxcւ  �          @���n{�33���R�mCx�n{�L����G�C7�                                    Bxc֐�  �          @�{�}p��(���=q�s�RCu�ÿ}p�=�G����H��C-�                                    Bxc֟h  �          @�z�^�R�����  33Cr��^�R?.{����HC
                                    Bxc֮  �          @�G��aG��?\)�z�H�N(�C}�H�aG��J=q���
��C]��                                    Bxcּ�  �          @�p�����n�R�Y���$��C}�ÿ����Q���Q�8RCm)                                    Bxc��Z  �          @�ff�z�H�����*=q�G�C�5ÿz�H�z���Q��m��Cw
                                    Bxc��   "          @�{�&ff�z=q�<����C�C׿&ff��
��ff=qC|c�                                    Bxc��  �          @����ff��G�����ffC�׾�ff�0  �|���[  C�`                                     Bxc��L  �          @��H������ff������C��{�����#�
��33�gG�C��=                                    Bxc��  �          @�33��Q���p���(���C����Q��=p��tz��P�C��                                     Bxc��  �          @�33�\)��Q��/\)�{C�<)�\)�p����qQ�C���                                    Bxc�#>  �          @��\��Q����R�333�ffC�e��Q�������R�w=qC��\                                    Bxc�1�  �          @���>�=q�����@  �\)C��>�=q�Q�����ffC��H                                    Bxc�@�  �          @�
=?�\�w��Dz���\C��H?�\���H��G��)C�L�                                    Bxc�O0  �          @�  �#�
�h���6ff�(�C��R�#�
��{���u�C��                                     Bxc�]�  �          @�\)���R�e���� 33Cz�쿞�R���q��gG�Cnz�                                    Bxc�l|  �          @����(��dz��ff��
=Cw�H��(�����b�\�V��Ck��                                    Bxc�{"  �          @�p��\�S33�   �
=qCu8R�\��(��qG��jz�Cdz�                                    Bxc׉�  �          @�p���
=�Vff�{���Cv�쿷
=���
�p���j��Cg�                                    Bxcטn  �          @�������>�R�333��\Cn:���Ϳ�ff�xQ��oQ�CW�                                    Bxcק  T          @����\����9���$z�CbT{��\�=p��l(��cp�CE�                                    Bxc׵�  �          @���&ff����-p��C\=q�&ff�(���[��O�RCB(�                                    Bxc��`  �          @�z��;������?\)�/G�CH��;�>��H���:z�C*�q                                    Bxc��  �          @�Q�����9���!G��
=Cq���Ϳ�{�e�n��C\B�                                    Bxc��  
�          @�Q�    ��33��=q��ffC�f    �/\)�c�
�Qp�C�
=                                    Bxc��R  �          @����W
=�h�����  C�Y��W
=�{�dz��h  C�J=                                    Bxc���  �          @o\)��Q���R����$z�Ch�)��Q�O\)�Mp��o  CMxR                                    Bxc��  T          @xQ��(�� ���)���-  Ca�)��(���\�R�\�h�CB��                                    Bxc�D  �          @��׿��
�#33�
=q���CsQ쿣�
��(��G
=�q��C_��                                    Bxc�*�  �          @�\)���>�  �����{C0�����?�\)��ff��  C$�)                                    Bxc�9�  T          @����s33��Q������HC9\)�s33?O\)�����33C'�                                    Bxc�H6  T          @����Z�H��Q��������CG��Z�H<��#33��
C3�                                    Bxc�V�  �          @�
=�'
=��p��$z��ffCY.�'
=���L���I  C?O\                                    Bxc�e�  T          @�(��<�Ϳ��.{�)  C=��<��?W
=�(���"�
C$#�                                    Bxc�t(  �          @�p��<(�=u�9���2C2�q�<(�?�Q��!G��C�                                    Bxc؂�  �          @��;�?L���3�
�+(�C$�f�;�@�\����HC
                                    Bxcؑt  "          @��E�B�\�E�3��C7���E?�ff�3�
��C33                                    Bxcؠ  T          @�{�[���G��Y���.=qCD\)�[�?:�H�^{�2C'��                                    Bxcخ�  �          @����W
=��p��n{�8��CH
=�W
=?0���w
=�Ap�C(^�                                    Bxcؽf  �          @�33�a녿��>{�(�CJ  �a�>#�
�R�\�,�C1aH                                    Bxc��  �          @�z��j�H��{�2�\�33CK���j�H��Q��Mp��$�C5\)                                    Bxc�ڲ  �          @��
�j=q���+��
=CNc��j=q���
�N{�$��C9
=                                    Bxc��X  T          @�{�P�׿�Q��\)�z�COaH�P�׾����@  �)��C9J=                                    Bxc���  
�          @���@  ��ff�)���p�CS  �@  �����L(��:G�C:33                                    Bxc��  �          @�z��:=q��\)�,����CT���:=q��Q��P���@
=C;�                                    Bxc�J  �          @��R�B�\�G��#33��
CU���B�\�\)�Mp��8(�C>z�                                    Bxc�#�  T          @��\�<(�����G����CX���<(��\(��B�\�3G�CD:�                                    Bxc�2�  �          @��
�Mp����
���G�CQ��Mp���G��;��(��C;ٚ                                    Bxc�A<  �          @���U�������R��CM:��U����
�-p��  C9u�                                    Bxc�O�  �          @����P  ��������CI�)�P  <��1G��!�C3p�                                    Bxc�^�  �          @���U������p�CJ���U��\)�333�C5+�                                    Bxc�m.  �          @���X�ÿ�����\���CE� �X��>���!G��33C1��                                    Bxc�{�  �          @�33�j=q�W
=�������C@��j=q>��R����RC/33                                    Bxcيz  "          @���tz�O\)�p���C?�R�tz�>�{�����C.�                                    Bxcٙ   �          @�{��G����R��p���33C8p���G�?�Ϳ���z�C,B�                                    Bxc٧�  �          @��R�~{����p����
C5�
�~{?Tz��ff���C(0�                                    Bxcٶl  �          @����G�>\�У����C/  ��G�?�{��  �|��C%�\                                    Bxc��  �          @�p���z�>��������G�C0c���z�?�z�˅��Q�C$aH                                    Bxc�Ӹ  �          @����x�þ�ff�.�R�p�C:���x��?fff�'
=���C&�3                                    Bxc��^  �          @��\�vff�0���4z��{C>)�vff?=p��3�
�Q�C)&f                                    Bxc��  �          @�{�~{�+��6ff�G�C=�H�~{?E��5����C){                                    Bxc���  �          @�����(��Ǯ�8���33C9s3��(�?��
�.�R�z�C&
=                                    Bxc�P  �          @����\)�aG��*�H� ��C6����\)?�=q�����ffC%�3                                    Bxc��  �          @�������?   ��(����C-�R����?��׿�  ����C#\                                    Bxc�+�  �          @�{���?��
�������C$^����?��
�����Q�C��                                    Bxc�:B  �          @����33?�  ��z��T  C.��33@�B�\�
=C��                                    Bxc�H�  T          @�ff����?��Ϳ�G���C#\)����?�33�.{��p�C#�                                    Bxc�W�  �          @������?�=q���\z�C#�R���?�p��Ǯ���\CG�                                    Bxc�f4  �          @��\��Q�?��Ϳ�33�P��C$8R��Q�?�p���p�����C 
=                                    Bxc�t�  �          @�33����?�33����F=qC#������?�  �����Tz�C�)                                    Bxcڃ�  �          @�����?�=q�����F{C$�����?�Q쾨���n{C �f                                    Bxcڒ&  �          @�p���p�?�  �����V{C(xR��p�?��������
C#��                                    Bxcڠ�  �          @���p�?�\)��Q��TQ�C'#���p�?�ff��\���C"xR                                    Bxcگr  �          @����Q�?���ff�<  C �f��Q�?�(�����C�                                    Bxcھ  �          @�
=���?��R�fff�
=C#!H���?޸R���
�h��C ��                                    Bxc�̾  �          @�����=q?�ff�O\)���C%����=q?\���Ϳ�ffC#B�                                    Bxc��d  T          @�G���  ?�G��Y���  C#+���  ?޸R�����RC ޸                                    Bxc��
  �          @�Q���\)?��H��33�K33C&W
��\)?�{��G�����C"
=                                    Bxc���  �          @����?��\��  �\��C%�
��?��H�   ��ffC �H                                    Bxc�V  �          @������?����{�o\)C%L����?��
�z����HC 
                                    Bxc��  �          @�Q���ff?�(���G��^�RC&33��ff?������\C!Y�                                    Bxc�$�  T          @����  ?�{��\)�p(�C'u���  ?У׿.{��C!�                                    Bxc�3H  T          @��\��  ?�  ��ff���C(����  ?�\)�c�
�C"�                                    Bxc�A�  �          @�  ����?��׿�p����C'�����?ٙ��E���RC ޸                                    Bxc�P�  �          @��
��ff?�
=��p���(�C#�)��ff?��H��R����C\)                                    Bxc�_:  �          @��H��{?�  �����o
=C#���{?�(��   ���C8R                                    Bxc�m�  �          @����
=?��R����g�C#E��
=?����������C��                                    Bxc�|�  
�          @�z����?�  ��{�j{C#L����?�(����H��p�C�                                    Bxcۋ,  �          @��H���
@  ��=q�hQ�C����
@'
=���Ϳ�{C��                                    Bxcۙ�  T          @�z���
=@�ÿ����tz�CaH��
=@!녾.{��
=C�                                     Bxcۨx  
�          @����  ?�
=�����fffCٚ��  @��u�&ffC�f                                    Bxc۷  "          @��R��p�?L�Ϳ�(���33C*����p�?�33�h���!�C$#�                                    Bxc���  �          @�z����H?aG���
=��\)C)�R���H?����W
=��RC#T{                                    Bxc��j  �          @�z�����?k���=q��
=C)(�����?Ǯ�s33�*�HC"                                    Bxc��  
�          @��H��G�?\(��\���HC*B���G�?�p��n{� ��C#�{                                    Bxc��  �          @�(���{?�R��  �W�
C-(���{?�녿O\)�
ffC'�{                                    Bxc� \  
�          @�p�����?�����8��C.B�����?z�H�333��
=C)u�                                    Bxc�  "          @���=q>���G��+
=C/
��=q?aG��+����HC*��                                    Bxc��  �          @��
��  >�33��=q�8��C033��  ?Q녿G��C+&f                                    Bxc�,N  �          @�33���>W
=����3
=C1�����?.{�O\)��
C,�)                                    Bxc�:�  �          @����z��G���Q�����C5+���z�=��
��p����HC3�                                    Bxc�I�  T          @��������Ϳ�����C5)���>#�
��\���RC2+�                                    