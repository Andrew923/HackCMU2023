CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230301000000_e20230301235959_p20230302021533_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-02T02:15:33.373Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-01T00:00:00.000Z   time_coverage_end         2023-03-01T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxi��  �          @�Q����+�@�33B|{C������=q@J�HA�G�C�Y�                                    Bxi��f  �          @��R����p�@�z�B��3C�R�������@U�B��C�|)                                    Bxi��  �          @�  �0����R@���B�(�C|�3�0������@c�
B�C��                                    Bxi���  T          @�G��O\)��@��HB���Cy.�O\)��
=@j�HB  C�&f                                    Bxi��X  �          @�G��k��z�@��HB��fCv{�k���p�@l��BffC�q�                                    Bxi���  "          @�G��+����@�\)B�{Cy� �+���  @}p�B$�C��f                                    Bxi�פ  "          @�녿\)��z�@�
=B��{C}�q�\)���@y��B 
=C���                                    Bxi��J  
�          @��ÿ
=��@�ffB���C|�ÿ
=���@��HB"ffC�s3                                    Bxi���  �          @�
=�
=q��z�@�
=B�k�C{ٚ�
=q��Q�@��B,
=C���                                    Bxi��  "          @\���ÿ���@��B�\)C�%������(�@�{B.�\C��                                     Bxi�<  
�          @����z῝p�@�z�B�#�Ct�)�z����@���B:�C��                                    Bxi� �  �          @���^�R��=q@���B�.Cl�f�^�R��(�@��B4�\C��                                    Bxi�/�  T          @�
=�fff�}p�@�=qB�#�Cc�R�fff�w
=@��RBA�C�s3                                    Bxi�>.  
�          @��R�}p��E�@��\B�CY�
�}p��k�@��HBI33C~��                                    Bxi�L�  �          @��
�0�׿G�@���B�z�Cdh��0���j=q@���BJ\)C���                                    Bxi�[z  �          @��׿
=�B�\@�{B��RCh=q�
=�fff@�
=BK�C�U�                                    Bxi�j   
Z          @����Ϳc�
@��B�Q�C[������o\)@�ffBCG�C}��                                    Bxi�x�  �          @��R�
=��@���B�p�C`aH�
=�Z�H@�G�BRp�C�                                      Bxi��l  "          @�G��Q녿��@��RB�aHCU��Q��Z�H@�33BRG�C�C�                                    Bxi��  �          @�=q����   @�ffB��CM�=����XQ�@��
BRp�C|�
                                    Bxi���  �          @��H����u@�\)B�� C@}q����J�H@�G�B\\)C{p�                                    Bxi��^  T          @������
>u@�ffB�\C&�����
�/\)@�Q�Bn�
CyxR                                    Bxi��  T          @������׾���@�p�B�ǮCBǮ�����K�@�ffBY33Cz�                                    Bxi�Ъ  �          @�=q������@��B�ǮC:�{����C33@���B\�CvǮ                                    Bxi��P  
�          @��
��=��
@�{B��
C0�q���7�@��Bc��Cs�{                                    Bxi���  �          @����\)���
@��B�33C4녿�\)�>�R@���B`�HCu^�                                    Bxi���  "          @�Q����>��@�
=B��C/aH�����,��@�Q�B^��Cl                                      Bxi�B  
�          @��
�$z�?p��@�  B^\)C���$zῡG�@z=qBW{CN\                                    Bxi��  �          @�G��7�?�  @��B[\)C ���7���{@��RBT
=CMk�                                    Bxi�(�  �          @�
=�/\)?G�@�33Bc�C$
�/\)����@�(�BR\)CQ޸                                    Bxi�74  
(          @�  �,(�?s33@���BdffC �)�,(����H@�Q�BYffCP}q                                    Bxi�E�  �          @���%?��
@�\)Bb�\C^��%��ff@���B\z�CN��                                    Bxi�T�  �          @��
�33?\(�@�\)Bu{CaH�33�Ǯ@�G�Bd{CV@                                     Bxi�c&  
Z          @����Q�?Q�@�(�BuC ��Q��
=@���Ba��CW&f                                    Bxi�q�  T          @�=q���>�@�  B��C)E�����@���BYG�C^0�                                    Bxi��r  T          @�  ���>�33@��B|�C+�������R@�z�BS��C^B�                                    Bxi��  T          @����\)>\@�{B�(�C+Q��\)�@���BV��C_33                                    Bxi���  
�          @��\�   >��R@�  B��3C,���   ��@�p�BT�
C`+�                                    Bxi��d  �          @�z��  ?@  @��B�33C!���  ��p�@���Bg�C]W
                                    Bxi��
  �          @�ff�!�?:�H@���By�C#��!녿�(�@�ffB_{CY��                                    Bxi�ɰ  
�          @�=q�	��?aG�@��B���C�)�	����@�G�Bm�\C\�{                                    Bxi��V  T          @���G�?\(�@�33B�u�C�3�G����@�=qBp�C^�3                                    Bxi���  "          @��\��\)?���@�(�B��
C� ��\)��33@�\)B}��C]�=                                    Bxi���  �          @�{�˅?�(�@��B��
Ch��˅���@�
=B�  C`@                                     Bxi�H  �          @�G���=q?��@�33B�.C
�{��=q���@�(�B��C[T{                                    Bxi��  �          @��R���?�  @��By�Cp���׿���@�\)B���CQ                                    Bxi�!�  �          @���	��?�=q@�G�Bsp�C�)�	�����@�p�B�RCN�H                                    Bxi�0:  T          @�z����?��@��HBm(�C�f��׿p��@��\B��=CJ��                                    Bxi�>�  
�          @����\)?�{@��HBl
=CJ=�\)�aG�@��B���CIn                                    Bxi�M�  �          @��
�H@�@��HBi��C�)�
�H�8Q�@�ffB�z�CFk�                                    Bxi�\,  �          @��
���?��
@�Q�Bg��CO\��ÿfff@�  B{�
CH��                                    Bxi�j�  
�          @��\��=q?޸R@�B{��Cp���=q���\@��
B��\CQ.                                    Bxi�yx  T          @�G�?Y����ff@�  B�Q�C�^�?Y����z�@|(�B+�HC��\                                    Bxi��  T          @��\?#�
���H@�33B�G�C�ٚ?#�
���@�=qB1�C�l�                                    Bxi���  
�          @�=q�L�Ϳ�=q@�B��C��L���u�@��HBB=qC��H                                    Bxi��j  
�          @�녾\)���@��B��\C��
�\)�~�R@��B8�HC���                                    Bxi��  �          @�=q<��
���@��B�{C�T{<��
����@��
B5C�q                                    Bxi�¶  �          @�\)���n{@���B�G�Cp�Ϳ��g�@���BF�C���                                    Bxi��\  �          @�p��   ���@��RB��=C[ff�   �HQ�@�  B[G�C�z�                                    Bxi��  �          @�(�?O\)���@��\B�z�C�  ?O\)��
=@P��Bz�C�"�                                    Bxi��  
�          @�
=��\)�E�@���B�  CV�Ϳ�\)�U@���BFz�C{n                                    Bxi��N  T          @��\���\��ff@��B��\Ca�׿��\�W�@j=qB8G�C}8R                                    Bxi��  
�          @�����Ϳu@�p�B��C\ٚ�����Tz�@p  B<{C{�
                                    Bxi��  �          @�{���u@�z�B�L�CV  ���S�
@n�RB7��Cv��                                    Bxi�)@  �          @�=q��  ��z�@��B�
=CU�{��  �U@X��B'��CrY�                                    Bxi�7�  �          @�
=��G����\@��B�Q�Cg�쿁G��e@eB/�C~@                                     Bxi�F�  "          @��?\(���=q@��B�33C�9�?\(��h��@��\B=��C��)                                    Bxi�U2  "          @�(�?��
���R@�(�B���C���?��
�r�\@�Q�B6ffC��3                                    Bxi�c�  
�          @�=q?�zῐ��@���B���C��?�z��h��@~�RB8��C��)                                    Bxi�r~  
           @�ff>�\)�0��@��HB�.C��>�\)�R�\@�Q�BP�
C�j=                                    Bxi��$  *          @�(��#�
��  @�\)B�aHC���#�
�hQ�@�\)BEp�C��R                                    Bxi���  �          @�녿333�G�@�p�B�(�Cd)�333�Z=q@�G�BKC�5�                                    Bxi��p  �          @��׿J=q�u@�B��\CD녿J=q�Dz�@���Ba�HC��                                    Bxi��  �          @�ff��\)�@  @��
B���Cy�)��\)�^�R@�\)BP33C���                                    Bxi���  �          @�
=��녿�G�@��RB��C~޸����{�@p  B-�C�
=                                    Bxi��b  T          @�  �@  ���@���B�z�CXh��@  �L��@�z�BT�
C�j=                                    Bxi��  "          @��׽�Q���@�\)B�\)C�zὸQ��N�R@��RBX�C�8R                                    Bxi��  "          @�Q�aG���@�ffB�
=Cw��aG��L(�@�{BX��C�H                                    Bxi��T  �          @��\����@�Q�B�ffCcn���@��@���BS�C�                                    Bxi��  
�          @��
��\>�(�@�G�B�=qC���\��\@��B���C�                                    Bxi��  T          @�33���
���@g�Bz=qC��
���
�\(�@(�B��C�]q                                    Bxi�"F  "          @��>Ǯ�P  @>{B(�RC�g�>Ǯ����?p��AC�C��{                                    Bxi�0�  "          @�z�>L�Ϳ�p�@�  B�\)C�Y�>L���hQ�@3�
B��C���                                    Bxi�?�  T          @��>��R�A�@�(�BV�C���>��R��{@�
A�\)C�޸                                    Bxi�N8  "          @�G�?�=q����@E�B{C�}q?�=q����?#�
@��C�\                                    Bxi�\�  �          @��
?}p���z�@P  B��C���?}p����R?@  A z�C�aH                                    Bxi�k�  �          @��?�=q����@5�A�(�C��
?�=q��(�>��R@P��C��R                                    Bxi�z*  �          @��R@����?���A���C���@���(��.{���
C��                                    Bxi���  �          @��@����z�?У�A��HC���@�����\�fff�\)C�:�                                    Bxi��v  �          @�\)@����ff?˅A��C���@������xQ��#\)C�s3                                    Bxi��  "          @�(�?�����Q�?��
A���C�#�?�����z῅��333C�޸                                    Bxi���  �          @�(�?��
��
=@�A��C���?��
��33�����z�C�W
                                    Bxi��h  T          @���?}p�����@   A���C��q?}p���z�W
=���C�7
                                    Bxi��  
�          @���>����@B�\BQ�C��
>����>���@�Q�C���                                    Bxi��  �          @��׾�������@c�
B"ffC���������
?��
A-��C�ff                                    Bxi��Z  "          @��R�����H@=qA�RC�������R��\)�J=qC�>�                                    Bxi��   L          @��\?���p�?��HA�(�C�:�?����ͿTz����C�3                                    Bxi��  *          @��?���?��AM��C��\?���(���=q�o33C���                                    Bxi�L  
(          @�p�?����n�R@'�B ��C�1�?�����G�>�
=@�ffC�R                                    Bxi�)�  L          @�Q�.{��\)@�{B��Cz.�.{�l��@-p�B
=C��{                                    Bxi�8�  
�          @�=q���R�P��@tz�BEffC�B����R��
=?ٙ�A�Q�C��                                    Bxi�G>  �          @��;B�\���@�A���C��3�B�\��Q�����C��=                                    Bxi�U�  �          @�\)����\)?��AO
=C�:�����Ϳ�(���33C�7
                                    Bxi�d�  
�          @�  �\(����>��R@UC�E�\(���z�����z�C��                                     Bxi�s0  
�          @��;������?���AL(�C��q�������Ϳ�33���
C��3                                    Bxi���  �          @�z�?�z��E@n{B:p�C��?�z�����?��HA�(�C��                                    Bxi��|  
�          @��?��R�Q�@a�B1C�5�?��R���H?��HA�z�C���                                    Bxi��"  T          @�  ?
=�h��@�A�p�C�� ?
=��G�>B�\@ ��C��\                                    Bxi���  
�          @�  >���  ?�ffA�33C��H>����׿E��
�RC���                                    Bxi��n  
�          @�p��#�
��Q�@�\A���C��3�#�
��{��G����C��{                                    Bxi��  �          @�\)�Y����\)@1�B=qC�XR�Y�����>�33@w�C�>�                                    Bxi�ٺ  
�          @�  ���R�<(�@xQ�BFz�Cs:῾�R��
=?�Q�A��RC{�)                                    Bxi��`  
�          @��ÿQ��fff@�=qB>��C����Q���z�?��
A��C�q�                                    Bxi��  T          @�ff���R���@��
Bn=qCn�q���R����@C33B��C{��                                    Bxi��  �          @���(���33@�HA���Cu�
��(���\)���
�L��Cx\)                                    Bxi�R  �          @��Ϳ��
��(�?�=qAe��Cy����
�����p��S\)Cz
=                                    Bxi�"�  "          @�33�p����\?J=qAp�Cv��p���=q������Cuc�                                    Bxi�1�  �          @�(��������@+�A؏\Cx������H�L�;��C{T{                                    Bxi�@D  
�          @�p��z�����@0��A�p�Cv�\�z�����=��
?J=qCyk�                                    Bxi�N�  "          @�p��  ���R@1G�A�p�Ct���  ���R=�?�
=Cw�f                                    Bxi�]�  �          @�p��{���@0��A�=qCt��{���=���?z�HCw�R                                    Bxi�l6  �          @�ff�{����@!�A��Cr���{�����G����Cus3                                    Bxi�z�  "          @�33�(����G�@��B>ffC�]q�(�����R?��HA�Q�C��{                                    Bxi���  
�          @�{>�
=����@��BB��C��)>�
=��Q�@z�A�C�
                                    Bxi��(  T          @�ff?}p��@  @�=qBm
=C��?}p���ff@L(�A��\C�`                                     Bxi���  �          @���?s33�7
=@��HBr33C�'�?s33��33@R�\B�C�E                                    Bxi��t  T          @Å?&ff�@  @���BoG�C�  ?&ff��p�@J=qA��C���                                    Bxi��  �          @��H=u�Z�H@���B_33C�xR=u��@0  A���C�K�                                    Bxi���  "          @�����Q��g�@�33BT�C�Q콸Q���Q�@\)A¸RC��                                    Bxi��f  	�          @�녿���W�@�  B^ffC��q������@0  A��C��{                                    Bxi��  T          @�
=�&ff�J=q@�Q�Bd��C�1�&ff��{@7�A�  C�n                                    Bxi���  �          @����R�U�@�{B^Q�C�� ��R����@.{A�z�C��                                    Bxi�X  T          @�\)��G��J=q@���Bf�C����G���ff@9��A��HC���                                    Bxi��  T          @�p��G��<��@�=qBk�C��G�����@AG�A�C���                                    Bxi�*�  �          @��Ϳ�ff�(��@��Bt��CxG���ff����@P��B=qC�Ф                                    Bxi�9J  
Z          @��
���\�#�
@�Bw�Cx5ÿ��\��\)@S�
B	�\C��f                                    Bxi�G�  
�          @��
�xQ��\)@�\)B{Cx���xQ���{@X��BG�C�%                                    Bxi�V�  �          @�33����@�  B��Cv�����=q@_\)B�
C��R                                    Bxi�e<  �          @�(���z����@�=qB���Cr:῔z���\)@g�BG�C��                                    Bxi�s�  T          @��
��ff���
@�(�B��C`�q��ff�l��@��B9\)Cz�3                                    Bxi���  T          @��ÿ�녾W
=@�ffB�Q�C;aH����2�\@�{B]�\Co�)                                    Bxi��.  �          @�ff��Q쿆ff@�z�B�W
CX
��Q��`  @��B@�HCw�{                                    Bxi���  �          @��R��=q�B�\@���B�\)CW(���=q�S�
@���BQz�C|�                                    Bxi��z  �          @�33��p���=q@dz�B�Cw  ��p�����?���ABffC{޸                                    Bxi��   T          @���33��
=@.�RA�G�Cv�
�33��ff>��?�(�Cyp�                                    Bxi���  �          @�녾�׿n{@��B�8RCs
=����`��@�G�BPQ�C�,�                                    Bxi��l  �          @�=q�5�0  @��Bqz�CxR�5��G�@EB{C��                                    Bxi��  �          @���n{���H@w�B*��C��{�n{��{?��HAmp�C�!H                                    Bxi���  T          @������Z=q@��RBS
=C|)�������@!�AʸRC��                                    Bxi�^  �          @�ff�˅����@��B,��Cxn�˅��
=?�A��RC}Ǯ                                    Bxi�  �          @�ff��p��dz�@�Q�BE��Cw����p�����@�\A�G�C~^�                                    Bxi�#�  �          @�ff��33�U@�(�BL�\Cs�ÿ�33��z�@   A���C|(�                                    Bxi�2P  T          @����p��^{@��RBP  Cz����p�����@!G�A�
=C�w
                                    Bxi�@�  "          @�ff����N{@�=qBX�HCw�f������@.�RA�
=C��                                    Bxi�O�  �          @�ff����B�\@�{B`��Cw�����Q�@:�HA뙚C�\                                    Bxi�^B  �          @����\)�>�R@�
=BeffCyh���\)��
=@>�RA���C��f                                    Bxi�l�  "          @������ff@�Q�B|
=Cq33�����G�@b�\B\)C~&f                                    Bxi�{�  �          @�(��������@�  B}��Cv  �������H@`��BC�p�                                    Bxi��4  �          @�33������@�{B{�
Cts3�������@]p�B��C��                                    Bxi���  �          @�=q�8Q��E@�(�Bc��C�ff�8Q���Q�@7
=A뙚C���                                    Bxi���  �          @�녿��
��  @~�RB/��C�쿃�
����?�33A�p�C���                                    Bxi��&  "          @������
���@{�B*C|�Ϳ��
��?���A|  C�aH                                    Bxi���  
�          @��\��Q�����@fffB��C�����Q���{?���A)�C�0�                                    Bxi��r  �          @��H��z�����@j=qB\)C.��z���33?�Q�A;�
C�!H                                    Bxi��  �          @�33��G�����@qG�B!  C}�Ϳ�G���G�?���AUG�C��{                                    Bxi��  "          @�녿n{���@uB%�
C�Ǯ�n{��G�?�Ac33C�5�                                    Bxi��d  �          @���������z�@S33B
�HC�� ������z�?B�\@�\C��                                     Bxi�
  T          @�녿�{��\)@\(�B�C}쿮{���?uA��C�q                                    Bxi��  
�          @�33�˅��z�@`��B��Cz
=�˅��Q�?���A)��C}��                                    Bxi�+V  
�          @���ff��=q@VffBz�Ct��ff��(�?uA�Cx��                                    Bxi�9�  T          @�z��
=���\@XQ�B=qCt��
=����?}p�A33Cx��                                    Bxi�H�  T          @��Ϳ�p�����@a�B��CuͿ�p���p�?�z�A5�Cy�H                                    Bxi�WH  "          @��(���@dz�B�\CrT{�(����?��RAAp�Cw�                                     Bxi�e�  T          @���"�\��(�@X��BCnn�"�\��\)?���A+\)Ct�                                    Bxi�t�  �          @�(��%��z=q@c�
B�Cl���%����?�{AU��Cs0�                                    Bxi��:  �          @��
�(���s�
@fffB=qCkT{�(����G�?�Q�Ac33CraH                                    Bxi���  T          @�����u@r�\B�Cms3�������?���AzffCt��                                    Bxi���  T          @��!��r�\@s�
B�ClL��!����?��A���Cs��                                    Bxi��,  
Z          @�p��&ff�n{@r�\BG�Ck��&ff��G�?�z�A�(�Cr                                    Bxi���  �          @��@  �W
=@uB!��Cd@ �@  ��  ?�\)A�\)Cm�3                                    Bxi��x  
�          @��$z��\)@c33B=qCm@ �$z���p�?�=qAO�Cs�)                                    Bxi��  �          @����*=q�~{@]p�B�RCl0��*=q���?��\AF=qCr��                                    Bxi���  �          @��
�-p��\)@VffB33Ck�\�-p����\?�z�A6�\Cq�                                    Bxi��j  �          @��Ϳ��R�}p�@y��B%Q�CsQ���R����?�z�A�
=Cyk�                                    Bxi�  
�          @�{�����@n�RB�HCqǮ�����?�(�AeCw�3                                    Bxi��  T          @�{����
=@X��BQ�Cp������?���A*�HCuT{                                    Bxi�$\  T          @��
�*=q��p�@G
=B CmxR�*=q���
?aG�A
�\Cr�
                                    Bxi�3  �          @�����q�@�ffB5�Cv#׿�����@�A�33C|k�                                    Bxi�A�  �          @�z��Q��tz�@���B3G�Cv{��Q���G�?�(�A�ffC|:�                                    Bxi�PN  T          @�ff�����o\)@��HB;33Cv�H��������@
�HA�G�C}8R                                    Bxi�^�  �          @����G��y��@�B3CxǮ��G����
?��HA�
=C~=q                                    Bxi�m�  �          @�p���(��~�R@�(�B0�HCy�=��(���p�?��A��RC~�)                                    Bxi�|@  "          @��Ϳ����x��@�{B533Cy���������?��RA��C~�f                                    Bxi���  �          @��Ϳ�G��y��@���B3{Cx�)��G���33?���A�z�C~B�                                    Bxi���  T          @�(��\�w
=@�p�B4��Cx���\��=q?��RA�{C~�                                    Bxi��2  
�          @����Q��q�@�z�B3��Cu녿�Q����@   A��
C|�                                    Bxi���  T          @�z��(��l(�@��B8p�Cu  ��(���ff@Q�A��C{�                                    Bxi��~  
�          @�33�޸R�a�@��\B>��Cs�R�޸R���H@�\A�ffC{�                                    Bxi��$  T          @�����Q��e�@�\)B;z�Ct���Q���33@�A��RC{��                                    Bxi���  �          @�=q�޸R�n�R@�33B3��Cu��޸R��p�@   A�G�C{h�                                    Bxi��p  �          @�=q����j�H@�33B3=qCr�=������@�A��Cy�                                     Bxi�   T          @�����j=q@�ffB7Q�Cs��������@Q�A���Cz��                                    Bxi��  "          @�33�\)�`��@��HB1�HCmp��\)��
=@ffA���Cu�3                                    Bxi�b  T          @�=q��
�S33@��B?�RCm�q��
���
@��A���Cw�                                    Bxi�,  T          @�33��Q��L��@��\BN�
Cr=q��Q�����@,(�A�
=C{�                                    Bxi�:�  �          @�(���=q�a�@��\B=�RCr���=q���\@�
A���Cz.                                    Bxi�IT  �          @�33��
=�n�R@�=qB0ffCr����
=����@   A�Cyu�                                    Bxi�W�  �          @��H��
�k�@z=qB'��Cm�H��
����?��A��
CuQ�                                    Bxi�f�  T          @��H��p��@s�
B"�HCn
�����?�G�A�=qCu(�                                    Bxi�uF  T          @�G��(��h��@qG�B"�
Cl)�(���?��
A��Cs��                                    Bxi���  �          @�=q�
=�n{@{�B*\)Cp��
=���\?��A��CwxR                                    Bxi���  
�          @�녿���p  @���B0z�CtxR�������?�(�A��Cz�=                                    Bxi��8  
�          @�=q���g
=@��B1=qCp�������@z�A�z�Cw�=                                    Bxi���  
�          @�����mp�@y��B*{Cp�������?��A�{Cw��                                    Bxi���  
�          @�Q��(��l(�@u�B&�HCoY��(���  ?���A��CvY�                                    Bxi��*  
�          @�����b�\@uB(33Cl.�����?�z�A�\)Ct�                                    Bxi���  T          @�Q��(��r�\@~{B.��Cu�׿�(���z�?�A�C{��                                    Bxi��v  �          @�G��   �p  @h��B��ClG��   ���R?�33A�{Cs@                                     Bxi��  
Z          @�  �:�H�hQ�@�p�BA��C�N�:�H���\@�A��
C���                                    Bxi��  T          @�p���z��j=q@��BFQ�C�����z���p�@�\A��HC�g�                                    Bxi�h  �          @�\)��
=�U�@�(�BX\)C�k���
=��Q�@.{A�C��q                                    Bxi�%  T          @�\)���H�Z�H@��BSG�C�����H���@'�A�Q�C�7
                                    Bxi�3�  �          @����
�Tz�@��\BW�C�E���
��\)@+�A�Q�C�,�                                    Bxi�BZ  "          @�ff�����hQ�@�\)B@��C}{������33@  A�z�C��f                                    Bxi�Q   T          @�{��\�z�H@��HB8{C�J=��\����@   A��RC�@                                     Bxi�_�  "          @��R�����e�@�p�BK\)C��{������z�@(�A���C�O\                                    Bxi�nL  
�          @�{�:�H�c�
@��BH�RC�1�:�H��33@��A�=qC���                                    Bxi�|�  "          @�p��O\)�q�@�(�B;z�C��q�O\)��{@ffA�ffC��
                                    Bxi���  �          @��h���x��@���B5{C�n�h�����?�(�A��\C�
                                    Bxi��>  �          @�{�:�H�n{@�\)B@�C�q�:�H��p�@�RA��HC��)                                    Bxi���  �          @����  ���@fffB�RC}5ÿ�  ����?�p�Ar�\C�XR                                    Bxi���  �          @�p��+��|(�@�Q�B4\)C�1�+�����?�Q�A�=qC�e                                    Bxi��0  T          @�z�n{��G�@aG�B�C����n{��(�?�{A_33C��                                    Bxi���  �          @��
�����  @\(�B�
C})������?�ffAUG�C�+�                                    Bxi��|  
�          @���?\)�G�@�(�B^Q�C�
=?\)����@7
=A��C�L�                                    Bxi��"  
�          @�p�>�z��^{@�
=BP�C�k�>�z���G�@%�A�=qC��=                                    Bxi� �  
Z          @�>\)�U�@��HBX�C�4{>\)���R@/\)A�p�C��                                    Bxi�n  
�          @�33<��
�J�H@��B]�HC�%<��
���\@5�A�33C�R                                    Bxi�  
�          @���z��!�@�{By  C��׿z���z�@Z=qBG�C�5�                                    Bxi�,�  "          @���O\)�.{@��\Bn\)C}z�O\)��  @N�RB�HC���                                    Bxi�;`  "          @�����J�H@��HB\\)C��R�����@4z�A�z�C�@                                     Bxi�J  �          @��H�(��I��@��B[ffC�}q�(�����@3�
A��
C�W
                                    Bxi�X�  T          @��H�\)�N{@���BX��C�ÿ\)���\@0  A�=qC��{                                    Bxi�gR  �          @���(���\(�@�BNz�C���(�����@$z�AظRC�7
                                    Bxi�u�  "          @�(�����l(�@�ffBA\)C�aH������@G�A�=qC��f                                    Bxi���            @�=q�+��a�@�Q�BGffC��ÿ+���\)@��A�G�C�%                                    Bxi��D  
�          @��þ#�
�|��@w
=B133C��\�#�
��ff?���A���C�R                                    Bxi���  �          @���#�
�q�@~{B9�C����#�
���H@�A�  C�˅                                    Bxi���  
�          @��R>#�
�z�H@qG�B/ffC�1�>#�
��(�?��A��C��=                                    Bxi��6  T          @�ff=�G����
@Mp�B(�C���=�G����?���A;\)C���                                    Bxi���  
�          @��R=�\)���\@<��B�C�l�=�\)��z�?G�A�
C�\)                                    Bxi�܂  �          @�ff>aG�����@^{Bz�C���>aG���
=?�Q�Av=qC�<)                                    Bxi��(  
�          @�\)>��
��{@^�RBffC�4{>��
��Q�?�
=As
=C�                                    Bxi���  �          @�ff>�ff����@2�\A��HC��=>�ff��z�?�R@���C�h�                                    Bxi�t            @�{>�  ���R@FffB  C��R>�  ���H?z�HA&=qC�T{                                    Bxi�  
�          @�{>�������@>�RB=qC�R>�����33?Y��A(�C��                                    Bxi�%�  �          @�{>�33���\@:=qBp�C�+�>�33���
?E�AC�ٚ                                    Bxi�4f  �          @�\)>�(����@N{B�C��=>�(����H?�\)A=C�N                                    Bxi�C  "          @�  ?�p����R@L(�B{C���?�p�����?�
=AFffC��                                    Bxi�Q�  �          @�
=?�=q����@\��B��C�  ?�=q���\?�  A~�RC�P�                                    Bxi�`X  �          @��@�R�HQ�@h��B)�\C�.@�R���H@�\A��
C��
                                    Bxi�n�  �          @�p�@7
=�7�@e�B%�C�j=@7
=���H@ffA�z�C�z�                                    Bxi�}�  T          @�
=@(Q��B�\@mp�B*�HC�l�@(Q�����@	��A��C��                                     Bxi��J  "          @�Q�@�E�@xQ�B4ffC��
@����@�\A�C�H                                    Bxi���  �          @�  @(��Q�@r�\B/�C��R@(���G�@Q�A�p�C��q                                    Bxi���  T          @�
=?�=q�k�@fffB$�C�9�?�=q���\?�ffA�C�c�                                    Bxi��<  T          @�{?����|(�@^�RB��C�|)?�����Q�?˅A�(�C��\                                    Bxi���  �          @�{@33�n{@XQ�B{C�o\@33����?˅A�{C���                                    Bxi�Ո  
�          @�p�?�ff�vff@\��B
=C�H?�ff���?�{A�ffC�Ǯ                                    Bxi��.  �          @�?����\)@6ffBG�C��q?������?fffA�
C��                                    Bxi���  "          @�?�Q���Q�@p�A�=qC�H�?�Q����>�G�@��
C�&f                                    Bxi�z  
�          @�p�?������@(�A�ffC�}q?������>���@�\)C�u�                                    Bxi�   
�          @�{@\)�u@FffB\)C�(�@\)��Q�?�ffA]p�C��H                                    Bxi��  "          @�p�@p��Vff@^{BG�C�%@p����R?�A�z�C�s3                                    Bxi�-l  T          @�{@��Tz�@c33B#33C�q@����R?�
=A�=qC�K�                                    Bxi�<  
�          @�{@��U@c�
B$  C���@���\)?�
=A���C��                                    Bxi�J�  
�          @�@*=q�@��@h��B(�C���@*=q���R@Q�A�p�C�!H                                    Bxi�Y^  �          @�ff@Q��vff@Mp�BG�C�t{@Q����?�z�Ap  C��                                    Bxi�h  T          @�\)@   �O\)@dz�B$\)C���@   ��z�?��RA��C���                                    Bxi�v�  T          @�\)@#�
�J�H@j�HB(
=C�xR@#�
���@
=A�\)C�33                                    Bxi��P  
Z          @�
=@\)�U@a�B!�C�\)@\)��
=?�A���C��{                                    Bxi���  
�          @�\)@,(��$z�@~�RB;C�"�@,(��{�@(��A��C�+�                                    Bxi���  �          @�
=@+�� ��@���B?=qC�t{@+��y��@.�RA��C�>�                                    Bxi��B  �          @�\)@��>{@|��B8z�C�L�@���G�@p�Aՙ�C�y�                                    Bxi���  �          @��@G��6ff@��BA��C�Ff@G���\)@*=qA�p�C��                                    Bxi�Ύ  "          @�Q�@Q��.�R@��BC�C��@Q���z�@0  A��
C��                                    Bxi��4  
(          @�Q�@=q�'
=@�ffBGG�C�^�@=q��G�@5A�33C�b�                                    Bxi���  
�          @�Q�@��!�@���BK��C��R@��\)@<(�Bp�C�XR                                    Bxi���  T          @��@����@���B^�RC�^�@��r�\@S�
B��C��\                                    Bxi�	&  �          @�Q�@Q��8Q�@�BEz�C�9�@Q�����@.{A�G�C�7
                                    Bxi��  �          @���@��?\)@�  B:33C��@����@!G�A��
C�+�                                    Bxi�&r  
Z          @��?�(��>�R@�  BH  C���?�(���z�@0��A���C�                                    Bxi�5  
�          @��\@G��L(�@��HB==qC�#�@G�����@"�\AظRC���                                    Bxi�C�  �          @��@
=q�\(�@q�B+�C��@
=q��z�@	��A��\C�|)                                    Bxi�Rd  N          @��H@���Tz�@|��B3�HC�h�@�����H@
=A�C���                                    Bxi�a
  (          @�=q@
=�a�@o\)B(��C�t{@
=��ff@A�Q�C�)                                    Bxi�o�  
�          @�=q@ff�e@l��B&C�,�@ff��\)@�A���C���                                    Bxi�~V  �          @���?��R�b�\@qG�B+z�C���?��R��
=@�A��C�l�                                    Bxi���  "          @���?�33�[�@���B:\)C��)?�33��
=@��AͅC��=                                    Bxi���  "          @��?޸R�hQ�@tz�B.  C��
?޸R���@��A�p�C���                                    Bxi��H  "          @�=q@��s�
@\(�B{C��\@����H?�(�A��\C�ٚ                                    Bxi���  
�          @��H@p��{�@Q�B\)C���@p���(�?��A�  C�(�                                    Bxi�ǔ  "          @��H@!G���(�@2�\A�  C��3@!G���(�?�G�A&�RC���                                    Bxi��:  �          @�33@   ���\@!�A�ffC���@   ��
=?0��@�=qC�XR                                    Bxi���  T          @��H@"�\����@�A��HC�]q@"�\���
?!G�@љ�C��H                                    Bxi��  �          @��H@33��\)@4z�A��C��@33��\)?�G�A)�C�1�                                    Bxi�,  "          @���@\)�x��@C33B�RC�S3@\)��  ?���Ab=qC��{                                    Bxi��  �          @�G�?�
=��ff@-p�A�33C��)?�
=��z�?Tz�A��C�L�                                    Bxi�x  
�          @�G���R��@��B}�HCaH��R��G�@i��B'  C���                                   Bxi�.  T          @�  ��{�7
=@�G�B_��Cx޸��{���@HQ�B�C�                                   Bxi�<�  T          @���>�ff�A�@��\Ba�C�C�>�ff����@G
=B	�\C��q                                    Bxi�Kj  
�          @��H?c�
�P��@�BR�HC���?c�
��@8��A�  C�e                                    Bxi�Z  T          @��\?s33�r�\@~�RB6=qC��?s33��  @�A��RC�b�                                    Bxi�h�  
�          @�G�?E��{�@s�
B.Q�C���?E���=q@z�A��C�L�                                    Bxi�w\  �          @�Q�?��u@y��B4�C�33?�����@(�A��C�9�                                    Bxi��  
�          @��H>B�\�j=q@��RBC�
C�z�>B�\���R@#33A��C�R                                    Bxi���  
�          @�p�?5�i��@���BD  C�� ?5��
=@'
=A��C�                                    Bxi��N  �          @�z�?E��fff@���BD��C��?E���@(Q�A�Q�C�s3                                    Bxi���  �          @�(�?��b�\@��\BIz�C��)?���z�@-p�A���C�Y�                                    Bxi���  "          @�33?
=q�h��@�
=BC��C�1�?
=q��{@%�AۅC��                                    Bxi��@  �          @��H?B�\�j=q@���B@{C��f?B�\��@ ��A��
C�g�                                    Bxi���  �          @��H?J=q�p  @��B:p�C��3?J=q��\)@��A��HC���                                    Bxi��  
�          @�(�?�ff�z�H@n{B%�
C���?�ff��Q�@G�A���C���                                    Bxi��2  
�          @��?��
�y��@xQ�B-Q�C��?��
����@
�HA�33C��                                    Bxi�	�  �          @�33?�\)�n�R@vffB-C��
?�\)��(�@{A���C�*=                                    Bxi�~  �          @���?�\)�g
=@���B6=qC�3?�\)���\@�A�ffC�B�                                    Bxi�'$  Z          @��H?˅�l(�@w�B/�RC���?˅���H@��A�
=C�{                                    Bxi�5�  T          @�=q?���s33@o\)B)p�C��q?�����@ffA��
C���                                    Bxi�Dp  �          @���?����qG�@n�RB)\)C���?������
@
=A���C��                                    Bxi�S  �          @�33?����l(�@xQ�B/�HC��3?�����33@�A�z�C��                                    Bxi�a�  �          @��@ ���mp�@k�B$��C�=q@ ����G�@A�C�c�                                    Bxi�pb  
�          @�=q@=q�j=q@X��B�C��\@=q��z�?�=qA��RC��q                                    Bxi�  "          @�ff@�|��@a�B\)C��\@��ff?�\)A��C�l�                                    Bxi���  T          @��?��H�n{@y��B+33C��)?��H��(�@33A���C��\                                    Bxi��T  �          @�\)@
�H�g�@w
=B)�C�}q@
�H��Q�@33A�C�B�                                    Bxi���  "          @��@Q��[�@|(�B-p�C�` @Q����@��A�(�C���                                    Bxi���  
�          @��@	���Q�@�B;=qC��H@	�����@.�RA�p�C���                                    Bxi��F  �          @��R?�\�n�R@y��B-�\C��=?�\���
@�
A�(�C��{                                    Bxi���  
�          @��R?������@P  B
��C���?�����33?�Q�Aip�C�@                                     Bxi��  "          @�?�33����@]p�B�C�\?�33��
=?�(�A�\)C�}q                                    Bxi��8  �          @�?�
=��{@=p�A��C��?�
=��?���A3\)C�%                                    Bxi��  
(          @���?����u�@�Q�B0ffC�"�?�����Q�@��A�=qC��{                                    Bxi��  �          @�  ?����}p�@y��B+=qC��?������\@  A���C��
                                    Bxi� *  
�          @�\)?
=����@�Q�B2�\C�/\?
=��p�@�A��C�B�                                    Bxi�.�  T          @�  ?@  �x��@�z�B9(�C�q�?@  ���H@   A�p�C�,�                                    Bxi�=v  �          @�\)?�ff�n�R@�
=B=��C�� ?�ff��\)@(��A��C��R                                    Bxi�L  �          @�  ?Y���n{@���B@�HC�q�?Y�����@,��A�ffC�ٚ                                    Bxi�Z�  �          @�Q�?#�
�fff@�p�BI�C�\?#�
��p�@8Q�A���C��R                                    Bxi�ih  �          @�Q�@(��vff@j=qBp�C��
@(���(�@A�ffC��                                    Bxi�x  T          @�G�@)���c33@n{B �C�Z�@)�����
@  A���C���                                    Bxi���  �          @��@���s33@hQ�B�HC�j=@�����\@�A��
C�xR                                    Bxi��Z  T          @���@
=�n�R@y��B)ffC��q@
=��33@
=A���C��)                                    Bxi��   T          @�  ?�p��i��@��\B5�C��3?�p����H@#�
A��C�ٚ                                    Bxi���  T          @�
=?����n�R@��
B8�C�?�����p�@$z�A�Q�C���                                    Bxi��L  �          @�  ?�ff�g�@��\B5  C�8R?�ff���@%�A�33C�C�                                    Bxi���  
�          @��?��o\)@�Q�B1��C�?�����@�RA̸RC�p�                                    Bxi�ޘ  "          @�Q�?�Q��Z=q@�
=B<=qC��R?�Q�����@2�\A�\)C�U�                                    Bxi��>  �          @���@{�R�\@�ffB:p�C���@{����@3�
A���C��                                    Bxi���  �          @���@(��c33@\)B.��C���@(���ff@!�A��
C�}q                                    Bxi�
�  �          @���@���e�@�  B.ffC��H@�����@"�\A�33C�p�                                    Bxi�0  �          @���@J=q�l��@B�\B   C�<)@J=q����?˅A���C�z�                                    Bxi�'�  "          @���@��r�\@x��B(ffC�\@���z�@
=A�=qC�AH                                    Bxi�6|  T          @��?Ǯ�o\)@���B6�\C�U�?Ǯ��@(Q�A�p�C�˅                                    Bxi�E"  
�          @�=q?E��QG�@��BZ�C���?E���@VffB��C���                                    Bxi�S�  �          @���?�{�a�@�{BH�\C���?�{���\@>�RA�  C�w
                                    Bxi�bn  "          @�33?�ff�`  @���BM  C�T{?�ff���\@FffB Q�C�)                                    Bxi�q  
�          @�33@\)�n�R@mp�B!�HC���@\)����@\)A�{C��)                                    Bxi��  
�          @�33@~�R�~�R?��At(�C��H@~�R����=�G�?��
C���                                    Bxi��`  �          @��
@i����  ?�p�A���C�L�@i�����H>L��?�p�C�:�                                    Bxi��  "          @���@X������?��HA�
=C��@X������>�Q�@b�\C��q                                    Bxi���  "          @���@J�H��Q�@�A�\)C���@J�H��ff>�ff@��C�N                                    Bxi��R  �          @�(�@G
=��G�@ ��A��HC���@G
=��z�?n{AffC�AH                                    Bxi���  T          @��@L����(�@/\)AݮC��)@L�����?��HA<��C���                                    Bxi�מ  
�          @�
=@P����\)@(��A���C��3@P�����
?��A(  C��=                                    Bxi��D  �          @�\)@K���
=@.{A�(�C��@K���(�?�A5G�C���                                    Bxi���  
�          @���@G���(�@(��A�\)C��q@G���Q�?��A   C��
                                    Bxi��  T          @���@ ����p�@K�B 33C���@ ����\)?��An{C���                                    Bxi�6  
�          @���@���33@_\)B�C�3@���Q�?�{A�33C��\                                    Bxi� �  �          @���@(����33@N{BQ�C��H@(����p�?�{Aw�C���                                    Bxi�/�  �          @���@+���G�@e�B�C��f@+���  @33A���C�q                                    Bxi�>(  T          @�{@\)�s�
@�=qB1\)C�7
@\)����@5�A��C�                                    Bxi�L�  �          @�ff?���xQ�@�ffB7�C���?�����
@;�A��C��                                    Bxi�[t  �          @�
=?����X��@��
BN�
C�&f?�����G�@`  B	�C�h�                                    Bxi�j  �          @�
=@ff�l��@���B4ffC�0�@ff��@<��A��HC��R                                    Bxi�x�  T          @Ǯ@�
�g�@���B:{C�B�@�
����@FffA�ffC��q                                    Bxi��f  �          @�\)@G��mp�@��RB7  C��
@G����R@@��A�C�H�                                    Bxi��  T          @Ǯ@33�g
=@���B;
=C�AH@33��z�@HQ�A��HC��R                                    Bxi���  T          @Ǯ@�R�p  @��RB6�C�XR@�R��  @@��A�Q�C��                                    Bxi��X  �          @�G�@
=q�mp�@��HB;��C�)@
=q��  @I��A�(�C��3                                    Bxi���  �          @�=q?�
=�e�@�33BH  C�.?�
=��ff@[�Bz�C���                                    Bxi�Ф  �          @��?��s�
@�{B?�HC���?����
@Mp�A�\C��                                     Bxi��J  �          @ʏ\?���z�H@��
B;�C�=q?����ff@G
=A�=qC�z�                                    Bxi���  �          @�33@=q�}p�@�=qB,{C��@=q��z�@5�A��C���                                    Bxi���  T          @ȣ�?����g
=@�33BKp�C���?�����
=@\(�B�C��                                    Bxi�<  T          @ə�@	���r�\@���B9�C���@	����G�@FffA�=qC��f                                    Bxi��  �          @�=q@ff�tz�@�=qB9ffC�j=@ff���\@G�A�ffC�@                                     Bxi�(�            @ʏ\?�33�z=q@�33B:�C���?�33���@G�A�(�C�3                                    Bxi�7.  �          @ə�@
=��=q@��B �C��@
=��(�@�RA��C���                                    Bxi�E�  �          @ʏ\@   ��
=@~�RBQ�C�T{@   ��Q�@(�A��
C���                                    Bxi�Tz  "          @��@(�����\@o\)B��C��{@(����G�@�A��C�G�                                    Bxi�c   �          @�G�@<(���
=@g
=B=qC�h�@<(�����@A��C��)                                    Bxi�q�  T          @ə�@?\)��(�@Z=qB
=C�+�@?\)��\)?�{A�33C��                                    Bxi��l  T          @�=q@>{���R@VffA�C�˅@>{��G�?�\A�ffC���                                    Bxi��  T          @˅@C33���@Tz�A��C�{@C33����?޸RA~ffC��                                    Bxi���  "          @�(�@QG���\)@^{B�HC���@QG���33?�(�A�  C�O\                                    Bxi��^  �          @�@J�H��ff@mp�B�HC���@J�H����@{A�
=C�˅                                    Bxi��  T          @��
@�  ��  ?�33AJ�HC��R@�  ��\)=��
?:�HC�'�                                    Bxi�ɪ  �          @�(�@�=q�r�\@A�Q�C�%@�=q���H?�ffA33C�@                                     Bxi��P  �          @�p�@�Q��\)@
�HA��RC�@ @�Q���
=?Tz�@�p�C���                                    Bxi���  T          @�\)@��H����@{A�  C���@��H���?G�@޸RC�:�                                    Bxi���  �          @θR@��
��Q�@�A��C��R@��
���R?333@�{C�k�                                    Bxi�B  �          @�
=@����{@�A�Q�C���@����33?\)@�\)C��R                                    Bxi��  T          @�(�@������R?��A���C�w
@�����=q>��@l��C�\)                                    Bxi�!�  �          @�(�@�
=�p  @!�A�p�C��@�
=���H?�  A5�C��=                                    Bxi�04  "          @�33@�����=q@"�\A��RC�@������?�A)��C�ٚ                                    Bxi�>�  �          @�(�@��H�l(�@{A�z�C���@��H����?�p�A1G�C���                                    Bxi�M�  �          @�
=@����p  @�RA��C��@�����  ?z�HA�
C�%                                    Bxi�\&  �          @У�@����l��@�HA�z�C�"�@�����Q�?�
=A&ffC�%                                    Bxi�j�  �          @�G�@�\)���@\)A��C���@�\)���
?��A (�C��                                    Bxi�yr  �          @��@���|(�@)��A��
C�^�@�����?�=qA<z�C�G�                                    Bxi��  �          @��@����s33@7
=A�ffC��q@������?���A]��C���                                    Bxi���  T          @θR@�33�k�@*�HA���C��=@�33���?�
=ALQ�C�j=                                    Bxi��d  �          @�@��R�|��@=qA�=qC�AH@��R���?�{A�RC�h�                                    Bxi��
  �          @�p�@����}p�@  A�  C�g�@�����ff?s33AQ�C���                                    Bxi�°  �          @���@��q�@	��A��C���@���  ?k�A  C��H                                    Bxi��V  
�          @�(�@���z�H@�A�p�C��f@����(�?Y��@�(�C�{                                    Bxi���  
�          @�@�G���{@z�A���C��
@�G���{?uA��C�8R                                    Bxi��  !          @�@�33��(�@A�=qC�=q@�33��z�?�  AffC��3                                    Bxi��H  
(          @�  @�(�����@�A���C���@�(���\)?aG�@���C���                                    Bxi��  T          @У�@���Q�@Q�A���C��@���
=?W
=@�p�C�'�                                    Bxi��  
�          @У�@�p����@A��
C��)@�p����?8Q�@�=qC���                                    Bxi�):  
�          @У�@�ff���
@�A�
=C��q@�ff��z�?�ffA�
C��=                                    Bxi�7�  �          @�(�@�G�����@{A�
=C�N@�G����?Q�@�(�C���                                    Bxi�F�  O          @ƸR@y�����?���A���C�@y����G�?\)@�
=C��
                                    Bxi�U,  
�          @Ǯ@u��{@Q�A�\)C�p�@u���
?8Q�@�p�C�#�                                    Bxi�c�  
�          @�=q@�����H@�A�z�C�  @������?aG�A ��C�y�                                    Bxi�rx  
�          @ƸR@~{����@
=A�\)C�s3@~{��ff?E�@��HC�{                                    Bxi��  
�          @ʏ\@x����=q@�A�\)C�,�@x����
=?��@�p�C��                                    Bxi���  
�          @�G�@j�H����@33A���C�p�@j�H����?^�R@�C�3                                    Bxi��j  "          @�(�@w
=���@Q�A�G�C�\)@w
=���?xQ�A�C��H                                    Bxi��  �          @�(�@u���  @Q�A�p�C�/\@u���  ?uA\)C��R                                    Bxi���  �          @�\)@j=q����@5�A��C�� @j=q����?�33AG�
C��3                                    Bxi��\  �          @У�@aG���G�@EA�{C���@aG���  ?��Ah��C��                                    Bxi��  �          @�=q@]p���@B�\Aܣ�C�8R@]p����?�ffA[
=C�g�                                    Bxi��  �          @ҏ\@Z=q��Q�@>�RA��C�Ф@Z=q��p�?�p�AP(�C�R                                    Bxi��N  �          @ҏ\@g�����@Dz�Aޣ�C�@ @g����?У�Ae�C�O\                                    Bxi��  "          @Ӆ@c33��@A�A�z�C��
@c33��33?ǮAZ{C��                                    Bxi��  
�          @��
@g
=��p�@@  A׮C��3@g
=���H?��
AV=qC��                                    Bxi�"@  
�          @Ӆ@^{���\@8��A���C��R@^{���R?���A@��C�:�                                    Bxi�0�  T          @���@j�H����@AG�A�  C��@j�H��=q?ǮAYG�C�H�                                    Bxi�?�  
�          @�(�@dz���@C33A��C���@dz����?˅A]p�C��R                                    Bxi�N2  "          @���@c33����@J=qA��C���@c33���?ٙ�AmC��H                                    Bxi�\�  "          @�(�@fff��{@@  A��C��H@fff��33?��AV�HC��R                                    Bxi�k~  �          @�(�@�R��{@���B�C�U�@�R���@&ffA��C�8R                                    Bxi�z$  �          @�z�?�����Q�@�Q�B7  C���?�����p�@Q�A��C�T{                                    Bxi���  T          @�p�?�z�����@�z�B<�C��?�z���Q�@^�RA���C�33                                    Bxi��p  �          @�(�?8Q���{@�=qBH  C��?8Q���ff@k�B�C�Ǯ                                    Bxi��  �          @��?�33���R@�  B7Q�C�=q?�33���
@S33A���C�˅                                    Bxi���  "          @׮@G���
=@��B�C��@G����@   A�z�C�t{                                    Bxi��b  
Z          @�ff?n{��ff@�33BGp�C�9�?n{���R@n{BffC���                                    Bxi��  
�          @�?B�\���
@��RBL�C�5�?B�\���@uBffC��)                                    Bxi��  �          @�
=?n{����@��B9(�C�?n{��\)@X��A�z�C���                                    Bxi��T  �          @�>\���@�B>p�C�s3>\��@^�RA���C��\                                    Bxi���  �          @�=q?(����=q@�z�B@��C�]q?(����Q�@_\)B �RC�n                                    Bxi��  T          @Ӆ>k����@��BD��C���>k�����@fffB(�C�4{                                    Bxi�F  
�          @��ͽL���j=q@���Bbp�C��3�L�����@�=qB"  C���                                    Bxi�)�  
�          @�ff=����X��@�Q�Bn  C�Ф=�����33@��HB-�C���                                    Bxi�8�  
Z          @׮?z�H�p  @�ffBZ33C�W
?z�H���@�ffB�\C�p�                                    Bxi�G8  �          @׮?�  �]p�@�  B[33C�h�?�  ��33@��\B 33C��                                    Bxi�U�  
�          @أ�?�Q��x��@�  BLQ�C���?�Q���{@~�RBffC��                                    Bxi�d�  "          @��@*�H��33@�\)B%33C�@*�H���@G
=A�Q�C�q                                    Bxi�s*  
�          @ڏ\?����r�\@��HB[\)C��?�����p�@��\Bp�C��R                                    Bxi�  
�          @��H?s33�{�@���BW(�C��\?s33��G�@�\)B�HC��                                    Bxiv  �          @���?!G��_\)@���Bi�C�&f?!G���@�33B+(�C���                                    Bxi  "          @ٙ�?
=q�QG�@�Bs�C���?
=q����@�=qB4\)C�<)                                    Bxi­�  �          @��
?B�\��@��HB033C��H?B�\��  @H��A�G�C�Ǯ                                    Bxi¼h  �          @�z�?}p���@��\B;33C�P�?}p���=q@\��A��\C��                                    Bxi��  �          @��H?�\)��  @��HB=\)C���?�\)����@`��B C��                                    Bxi�ٴ  �          @�G�?�{���@�33B?��C�O\?�{��z�@aG�B\)C��=                                    Bxi��Z  �          @У�?(���y��@�BRffC��{?(�����@{�BQ�C���                                    Bxi��   T          @ə���z���\@��B�
=C�b���z��q�@���BSC��                                    Bxi��  �          @�(����Ǯ@�ffB���C>aH����@�z�BuffC\��                                    Bxi�L  4          @�(����n{@��
B�G�CK#����@�p�BcCc�                                    Bxi�"�  	`          @�  ���Ϳ�p�@�Q�B�p�CZ�������9��@��B[�HCmff                                    Bxi�1�  �          @�z῾�R�\@�  B�L�Ca�Ϳ��R�@��@�z�Bc=qCs�
                                    Bxi�@>  
�          @�=q��ff��=q@�  B��=Cl=q��ff�C�
@��
Bf��Cz��                                    Bxi�N�  
�          @�(�������\@��\B�.Ct�׿����l��@�Q�BL33C}��                                    Bxi�]�  
�          @�
=�h�ÿ�Q�@�=qB���Ct�ÿh���W
=@�33BY\)C~�)                                    Bxi�l0  	.          @�G��z�H��(�@��\B�{Cs��z�H�Tz�@��BTz�C}�=                                    Bxi�z�  
�          @�33��33��\@���B��fCb���33�G
=@�z�BS�Cr�                                    BxiÉ|  
�          @��Ϳ��H����@�p�B��3C^�H���H�<��@��BZ��Co�R                                    BxiØ"  
�          @�33>.{�AG�@�=qBq�\C���>.{���@��\B4G�C�"�                                    Bxiæ�  �          @�=q>B�\�@  @�Q�Bq(�C���>B�\����@���B4  C�H�                                    Bxiõn  
(          @�{?�=q�>{@���Bg=qC�H?�=q��@��B,C�=q                                    Bxi��  
(          @�\)?=p��Mp�@�
=Ba�HC���?=p����@|��B&  C���                                    Bxi�Һ  "          @�z�>���9��@��
BvG�C���>�����R@�p�B9p�C�
=                                    Bxi��`  �          @�p�>�z��`��@���B\z�C�` >�z����R@}p�B�
C��                                    Bxi��  T          @���?���_\)@�ffBV��C�Ff?�����@w
=B��C�E                                    Bxi���  	�          @��?���Z�H@�p�BR�C�� ?������@g
=B��C���                                    Bxi�R  �          @��?��H�/\)@��B\��C���?��H�qG�@\��B$Q�C��                                    Bxi��  �          @��R>������@��B�L�C���>����7
=@l��BP�RC��
                                    Bxi�*�  	�          @��@�H�:�H@(�A�(�C��@�H�X��?�{A�Q�C��                                     Bxi�9D  '          @�{@y���L(�����LQ�C�Y�@y���2�\��
=���C�9�                                    Bxi�G�  T          @�p�@vff�<�ͿB�\�=qC�Ff@vff�)����  ��{C��R                                    Bxi�V�  
�          @��@]p��:=q���
�}p�C���@]p���R���R�ȣ�C�*=                                    Bxi�e6  
�          @�p�@0  ����  ��C�Q�@0  ��ff�
=q�\)C�]q                                    Bxi�s�  
�          @��R@Vff��Q�>L��@	��C��@Vff��p��fff��C�c�                                    BxiĂ�  �          @�(�@�\��
=?�A���C��3@�\��G�?\)@�=qC�=q                                    Bxiđ(            @���?�=q��{@|(�B#(�C��R?�=q����@*�HA�{C���                                    Bxiğ�  �          @�ff?��
��{@s33BffC���?��
��Q�@"�\Aʏ\C���                                    BxiĮt  �          @��\?\(����@fffBC�q�?\(�����@�A���C���                                    BxiĽ  /          @�
=?\)��@h��B��C�l�?\)��ff@z�A��
C��                                    Bxi���  �          @�Q�?u���R@eB(�C��q?u��
=@G�A�{C��{                                    Bxi��f  /          @�G�>��H���@^�RB��C��{>��H��(�@
=A�{C�w
                                    Bxi��            @��H<#�
��p�@\��B
=C��<#�
��z�@	��A�=qC�3                                    Bxi���  /          @�(�?Q���ff@���B,�RC��?Q����\@4z�A噚C��{                                    Bxi�X  �          @��?c�
���
@�=qB/��C��?c�
����@8��A�\C�H                                    Bxi��  /          @�G�>Ǯ�u�@��RB>(�C��>Ǯ����@G
=BC�\)                                    Bxi�#�  �          @����s33�4z�@��Be�HC{k��s33�z=q@s33B-�\C�.                                    Bxi�2J  �          @�  �����C33@��B`
=C�=q������33@h��B%��C�33                                    Bxi�@�  4          @�녿��\�Mp�@�BR��C|\)���\��
=@^�RB�RC�33                                    Bxi�O�  4          @�z�� ���   @��
B\Cg8R� ���e�@uB,=qCp�3                                    Bxi�^<  �          @��� �׿�p�@�33Bi��C`�
� ���E�@|��B<=qClǮ                                   Bxi�l�  4          @�z��
=q�  @��BY��Cb��
=q�QG�@j=qB,33Cl��                                    Bxi�{�  �          @�33�@�׿��@��\B\�RCG��@���	��@�p�BACW��                                    BxiŊ.  �          @�Q��"�\��(�@�\)Br
=CM��"�\��H@�Q�BP�\C_��                                    BxiŘ�  4          @�G�������@��\Bt��C\�{���AG�@��RBI�\Cj��                                    Bxiŧz  f          @�G��0  ��@�{BWCX� �0  �H��@�G�B0Q�Cd��                                    BxiŶ   �          @�Q��%���@�G�B`�CW޸�%�@��@�p�B9�\Ce8R                                    Bxi���  4          @�p��,���G�@��BL�C\��,���Q�@j=qB"��Cf��                                    Bxi��l  �          @�  �fff���H@�33BAC;�q�fff���R@vffB2�\CJ}q                                    Bxi��  �          @������?Tz�@}p�B,G�C(������.{@�G�B0��C6c�                                    Bxi��            @�����?�G�@UB��C&�����>��@_\)B�C1�q                                    Bxi��^  4          @�z���33?�@\(�B
=C,�q��33��{@]p�B G�C8Ǯ                                    Bxi�  /          @������>���@`��B#��C/����׿�@^{B"{C<\                                    Bxi��  9          @�(��z�H>k�@g
=B*�\C0���z�H�+�@c�
B'\)C=�                                    Bxi�+P  �          @�{���?(�@{�B.�HC+h���녾�p�@}p�B0�\C98R                                    Bxi�9�  �          @��\)?c�
@y��B.�\C'p��\)��Q�@�  B4(�C5\)                                    Bxi�H�  /          @��H�Z=q���
@�=qBH�C5T{�Z=q��=q@{�B>�CEz�                                    Bxi�WB            @�p��j�H��@~�RB=33C5�)�j�H���@u�B4
=CD�=                                    Bxi�e�  �          @��H�i���#�
@vffB:{C4(��i���s33@n�RB2�RCB�
                                    Bxi�t�            @�{�c�
=u@��HBC�RC2�R�c�
�p��@~�RB<�CB��                                    Bxiƃ4  �          @�z��W��Tz�@��HBF�RCA���W���ff@qG�B2G�CP
                                    BxiƑ�            @��]p��z�H@��BB
=CC�{�]p���Q�@l��B,33CQJ=                                    BxiƠ�  /          @�Q��Z=q���H@z=qB6CN�f�Z=q�'
=@XQ�B�CYn                                    BxiƯ&  �          @�z��o\)��@`��BQ�CS�f�o\)�G
=@5A�\C[                                    Bxiƽ�  �          @�{�[�?�ff@|(�B9Q�C�[�?
=q@�ffBI��C+                                      Bxi��r  �          @����p�?��H@<(�A�C �{��p�?s33@QG�BffC)�                                    Bxi��  
�          @�p���?�G�@AG�A�G�C%�3��>��H@N�RBffC.Q�                                    Bxi��  �          @�
=��?Q�@K�B=qC*
��<�@Q�B=qC3��                                    Bxi��d  
�          @�p���{?�@C�
B	C-Y���{�u@FffB�\C7
=                                    Bxi�
  T          @�����>\@;�B�C/.��������@;�B��C8.                                    Bxi��  T          @�=q���
�#�
@EB  C5�3���
�fff@=p�B �C>��                                    Bxi�$V  "          @�z����
>�Q�@C�
Bz�C/T{���
�Ǯ@C33BG�C9\                                    Bxi�2�  T          @�33����?�\)@@  B�C#(�����?(�@O\)BffC,O\                                    Bxi�A�  �          @�����H?�  @I��B(�C'�����H>aG�@S33Bz�C1=q                                    Bxi�PH  �          @�����G���{@\(�BG�CA����G���33@Dz�A��CJ�                                    Bxi�^�  �          @������G�@a�BQ�C=���������@O\)B�CG)                                    Bxi�m�  �          @�\)��z�E�@l(�Bp�C=u���z�У�@Y��B
��CG^�                                    Bxi�|:  �          @�ff��z�L��@hQ�B�C=�q��z���@UB�CGp�                                    BxiǊ�  �          @�=q��Q�0��@l(�B�C<8R��Q��ff@[�B	�CF                                      BxiǙ�  T          @�G����
��p�@a�B��C8Q����
����@UB�
CA�\                                    BxiǨ,  �          @�����\�u@^{B	Q�C4����\�aG�@W
=B=qC=ٚ                                    BxiǶ�  "          @��
���þ�@^�RB
�HC5p����ÿs33@VffB
=C>��                                    Bxi��x  "          @�33��  <�@_\)B�HC3����  �J=q@Y��B�RC=�                                    Bxi��  T          @��
��z�<#�
@Tz�BffC3���z�G�@N�RA���C<��                                    Bxi���  T          @Å���=�@QG�B��C2�3����(��@Mp�A�p�C;J=                                    Bxi��j  
�          @�z����R>aG�@O\)A��C1�����R���@Mp�A�C9�q                                    Bxi�   �          @Å��{>k�@N{A�{C1u���{��@K�A���C9�H                                    Bxi��  T          @Å��>u@N{A�z�C1c�����@K�A�p�C9�\                                    Bxi�\  T          @ƸR����>��@Q�A�33C1+����׿�@P  A��\C9�H                                    Bxi�,  T          @���
=>�Q�@Q�B \)C0{��
=��
=@QG�B   C8�H                                    Bxi�:�  �          @Å��z�>��H@QG�B�\C.�{��zᾔz�@S33B�\C7G�                                    Bxi�IN  �          @�{��z�>��@[�B�C1#���z���@Y��B�C:&f                                    Bxi�W�  �          @�����>�ff@Y��BG�C/�����Q�@Z=qB�RC8{                                    Bxi�f�  
�          @�  ��  ?��@Q�Bp�C-�q��  �k�@Tz�B�C6��                                    Bxi�u@  �          @�����=q?:�H@I��B��C+W
��=q���
@O\)B��C4E                                    Bxiȃ�  
�          @�����Q�?=p�@P��B{C+����Q�#�
@UB��C4��                                    BxiȒ�  
�          @����=q?#�
@P  B��C,���=q��@S�
BQ�C5z�                                    Bxiȡ2  "          @Å��z�?!G�@O\)B 33C,�R��z�\)@S33B��C5��                                    Bxiȯ�  T          @�(���z�?5@P��B �C,(���z὏\)@U�B�
C4�                                    BxiȾ~  �          @�������?Y��@J�HA�{C*ff����=�Q�@Q�B{C2�                                    Bxi��$  "          @�ff��\)?h��@HQ�A��
C)����\)>#�
@P  B�C2&f                                    Bxi���  T          @�����  ?}p�@N{B�C(����  >aG�@W
=B�C1��                                    Bxi��p  T          @\��Q�?��
@R�\B
=C(W
��Q�>u@\(�B	�
C18R                                    Bxi��  
�          @\���?��@S�
B��C'�3���>�z�@^{B�C0�f                                    Bxi��  "          @��H����?��H@L(�A�Q�C&z�����>�G�@XQ�B�
C.��                                    Bxi�b  T          @Å��33?��@@  A�33C#.��33?O\)@Q�B��C+                                    Bxi�%  �          @�z����\?У�@B�\A�p�C"+����\?c�
@U�B��C*{                                    Bxi�3�  �          @���\)?Ǯ@8Q�Aޣ�C#ff��\)?Y��@J�HA���C*�q                                    Bxi�BT  �          @Å���?�{@6ffA�=qC"�����?h��@H��A��C*�                                    Bxi�P�  "          @�z���p�?���@9��A�G�C"�q��p�?fff@L(�A�Q�C*+�                                    Bxi�_�  
�          @\����?��@;�A�Q�C aH����?���@P��B�RC'��                                    Bxi�nF  T          @�=q���?��H@?\)A�(�C!\���?z�H@S33B�C(��                                    Bxi�|�  T          @��\���@G�@'
=A�33C���?�\)@@  B\)C#
=                                    Bxiɋ�  
�          @�����\)?��@   A�RC
��\)?�  @7
=B��C#}q                                    Bxiɚ8  �          @��H��G�>�=q@^{BC1
=��G���@\(�Bz�C9�3                                    Bxiɨ�  
�          @��H����?�  @@  A�z�C&W
����?�@Mp�A��\C.�                                    Bxiɷ�  �          @�33���?�z�@AG�A�{C'\)���>�G�@Mp�A�=qC/.                                    Bxi��*  �          @����ff@�@%�AŅC�{��ff?�(�@@  A�(�C$0�                                    Bxi���  
�          @���ff?s33@E�A�\)C)
��ff>k�@N{B��C1c�                                    Bxi��v  �          @�p���Q�?�(�@5A��C%����Q�?
=q@B�\B�\C-��                                    Bxi��  �          @�(���=q?Ǯ@p�AυC"���=q?u@0  A�=qC(�                                     Bxi� �  T          @�=q��z�?�33?���A�
=C�3��z�?�z�@A�p�C$�3                                    Bxi�h  �          @����G�?���@'
=A�Q�C"\)��G�?xQ�@:=qA�ffC)&f                                    Bxi�  �          @\���?��
@8��A�C#W
���?Tz�@J=qA�p�C*                                    Bxi�,�  �          @�(����@�H@�
A���C�R���?��@#33A�
=C�                                    Bxi�;Z  T          @�������@-p�?�A��RC������@{@=qA�
=C33                                    Bxi�J   �          @���33@B�\?ٙ�A�C5���33@%�@ffA�z�C8R                                    Bxi�X�  T          @�������@@  ?���A���CT{����@ ��@��A��C��                                    Bxi�gL  
Z          @�ff��33@?\)?�\)A�z�C����33@�R@   A�(�C�                                    Bxi�u�  �          @ə����@�@��A�
=CT{���?��H@:=qA�ffC"�                                    Bxiʄ�  
�          @ʏ\����?�p�@)��A�(�C�H����?�=q@A�A��HC&33                                    Bxiʓ>  T          @��
���R@ff@!G�A�=qC����R?�p�@;�A���C$�{                                    Bxiʡ�  
�          @���
=@��@$z�A�ffC���
=?���@@  Aޏ\C$                                    Bxiʰ�  
�          @�{��Q�@�@(�A�(�C� ��Q�?�@8��AՅC#!H                                    Bxiʿ0  
�          @�����R@  @�RA��
C�)���R?У�@:�HA��HC#Y�                                    Bxi���  �          @��
���@
�H@$z�A�(�C(����?��
@?\)A�  C$.                                    Bxi��|  T          @�33��{@	��@�RA�(�Cu���{?��
@9��Aٙ�C$E                                    Bxi��"  T          @�33��@�@ ��A�{C�3��?�  @:�HA��C$�{                                    Bxi���  
�          @ʏ\��{@@p�A�
=C����{?�p�@7�A׮C$�q                                    Bxi�n  
�          @��H��{@�@   A���C\��{?�(�@9��A��C$�                                    Bxi�  �          @�33��?��R@(Q�A�\)C޸��?���@@��A�  C&\                                    Bxi�%�  
�          @��
��
=?�
=@(Q�A���C ����
=?��@?\)A�z�C&��                                    Bxi�4`  �          @��H��p�?�z�@*�HA�p�C ����p�?�G�@A�A�RC&�f                                    Bxi�C  "          @ƸR���?�@(��A��C �)���?���@>�RA��C'=q                                    Bxi�Q�  
�          @������?�  @,��A�(�C!�\���?���@AG�A�p�C(.                                    Bxi�`R  �          @������?Ǯ@1G�A�=qC#������?c�
@C33A��
C*aH                                    Bxi�n�  T          @�33��{?��@:�HA�RC&  ��{?(�@HQ�A��C-T{                                    Bxi�}�  �          @������R?�ff@@  A�
=C%�3���R?��@Mp�A�C-s3                                    BxiˌD  �          @�����H?��@1�A��C �����H?���@HQ�A��
C'J=                                    Bxi˚�  �          @˅��(�@G�@.{A���Cff��(�?�{@G
=A��C%��                                    Bxi˩�  �          @�z���z�@
=@,��A�  C����z�?���@FffA�\C$�R                                    Bxi˸6  T          @������
@	��@.�RA�  C33���
?�p�@H��A�G�C$��                                    Bxi���  
Z          @�z���(�@G�@2�\A�G�Ck���(�?��@J�HA�=qC%��                                    Bxi�Ղ  �          @��
���@ ��@333A���Cp����?�=q@K�A�C&�                                    Bxi��(  "          @�(����
?��@8��A��C �)���
?���@O\)A�C'p�                                    Bxi���  �          @�(����
?�G�@=p�A�G�C!޸���
?�ff@Q�A��C(�                                    Bxi�t  T          @˅��33?���@B�\A�z�C#W
��33?aG�@Tz�A��C*�f                                    Bxi�  T          @�33��Q�?��R@N{A�{C$0���Q�?:�H@^{B�RC,{                                    Bxi��  �          @ʏ\���\?��
@C33A�\C#����\?O\)@Tz�A���C+Q�                                    Bxi�-f  T          @����33?\@?\)A�  C$)��33?Q�@P  A�{C+T{                                    Bxi�<  
�          @�G���(�?��R@8Q�A�ffC$����(�?O\)@H��A�  C+}q                                    Bxi�J�  
�          @�  ��G�?��R@>�RA�\)C$E��G�?J=q@N�RA��C+�                                    Bxi�YX  
�          @�Q�����?ٙ�@+�Aʣ�C"�\����?�ff@?\)A�=qC(�                                    Bxi�g�  T          @Ǯ��(�?޸R@(Q�A�C"
=��(�?�{@=p�A�Q�C(Q�                                    Bxi�v�  �          @�\)��z�?�  @$z�A���C!�q��z�?��@8��A݅C()                                    Bxi̅J  
�          @ƸR��(�?��
@"�\A�
=C!����(�?�@7�A�z�C'��                                    Bxi̓�  �          @�
=��z�?��
@!�A�=qC!� ��z�?�@7
=AۅC'Ǯ                                    Bxi̢�  �          @Ǯ����?�G�@%�A��C!�����?��@9��A�  C(�                                    Bxi̱<  T          @���33?��
@"�\A�  C!����33?�@7�A݅C'�f                                    Bxi̿�  �          @�ff���?��
@#�
A�\)C!�����?�z�@9��A޸RC'�                                     Bxi�Έ  
�          @����?�\@&ffAǅC!�=���?�33@<(�A��HC'��                                    Bxi��.  
�          @ƸR��G�?�  @/\)AхC!���G�?���@C�
A�ffC(L�                                    Bxi���  T          @�  ���\?�G�@.{A���C!�3���\?�{@C33A��
C(8R                                    Bxi��z  
�          @�  ��33?�G�@,��A̸RC!� ��33?�\)@A�A癚C(33                                    Bxi�	   
�          @�{���R?��@>{A�C#�����R?W
=@O\)A��RC*޸                                    Bxi��  
�          @�33���?�G�@8��A�Q�C#�����?Tz�@J=qA��C*޸                                    Bxi�&l  "          @\���\?�
=@A�A���C$L����\?8Q�@QG�B
=C+��                                    Bxi�5  �          @��H���\?���@C33A�RC$�)���\?(��@Q�Bz�C,�
                                    Bxi�C�  "          @�����Q�?�p�@J�HA��\C&8R��Q�>��H@W
=B
=C.ff                                    Bxi�R^  �          @�����?��
@UB
=C(B���>��@^�RB�C1�                                    Bxi�a  
�          @�
=��(�?�(�@Mp�B  C%���(�>�@Y��B
�
C.ff                                    Bxi�o�  �          @�  ��z�?�{@R�\B=qC'5���z�>�33@]p�B�
C/�                                    Bxi�~P  T          @�  ����?�p�@O\)B�RC%�����>��@[�B�\C.z�                                    Bxi͌�  �          @�\)��z�?�  @Mp�B�HC%����z�?   @Z=qB
�C..                                    Bxi͛�  �          @�Q���ff?��@I��A���C%n��ff?��@VffB�
C-�H                                    BxiͪB  "          @��R��{?���@C�
A��RC$����{?!G�@Q�B�C,��                                    Bxi͸�  "          @�\)���?��@I��A���C%{���?�@W�B	�C-\)                                    Bxi�ǎ  
�          @�
=��z�?�
=@FffA�ffC#����z�?5@UB33C+Ǯ                                    Bxi��4  �          @��R���R?��@@��A�\C$ٚ���R?#�
@N�RBffC,��                                    Bxi���  
�          @���{?�G�@9��A�RC#
=��{?Q�@J�HB33C*��                                    Bxi��  T          @�p���
=?�(�@7
=A���C#}q��
=?L��@G
=A��
C*޸                                    Bxi�&  "          @�{���?��R@7
=A�z�C#n���?O\)@G�A���C*�=                                    Bxi��  T          @��
��p�?��H@5A�  C#���p�?G�@FffA���C*�3                                    Bxi�r  
�          @������?��H@4z�A�Q�C#B����?L��@Dz�A��C*�3                                    Bxi�.  �          @��\���?Ǯ@:=qA���C"����?^�R@K�B(�C)�3                                    Bxi�<�  T          @�  ��  ?�
=@@  A��C$���  ?8Q�@O\)B�
C+                                    Bxi�Kd  
�          @�Q���Q�?���@@��A���C$����Q�?+�@P  B�RC,h�                                    Bxi�Z
  
�          @������?���@:�HA��C#�����?B�\@J�HA�\)C+c�                                    Bxi�h�  T          @�  ��Q�?��@?\)A��HC$s3��Q�?0��@N{B  C,&f                                    Bxi�wV  
�          @�
=����?�z�@8��A�C$O\����?:�H@HQ�A�
=C+��                                    Bxi΅�  �          @�
=��Q�?�\)@<(�A�  C$�3��Q�?.{@J�HB \)C,G�                                    BxiΔ�  �          @������?���@;�A�(�C%T{����?!G�@I��A��C,ٚ                                    BxiΣH  �          @�\)����?�  @=p�A��HC&�����?\)@J=qA��C-��                                    Bxiα�  �          @�{��Q�?�G�@;�A��C%ٚ��Q�?�@H��A�\)C-u�                                    Bxi���  T          @�{���?��H@?\)A�C&Q����?�\@K�B��C.)                                    Bxi��:  �          @��R����?�G�@<(�A��C%�)����?�@I��A�\)C-xR                                    Bxi���  
�          @�ff����?�{@4z�A���C$�R����?0��@C�
A��C,=q                                    Bxi��  
�          @�z���
=?���@7
=A�z�C%���
=?&ff@E�A���C,��                                    Bxi��,  �          @����?�=q@7�A��HC$���?&ff@FffA�\)C,}q                                    Bxi�	�  �          @�(���\)?���@5A�G�C%!H��\)?&ff@Dz�A�p�C,�\                                    Bxi�x  �          @�(����R?�ff@7�A�C%\)���R?�R@EA�p�C,�H                                    Bxi�'  �          @�(���\)?�  @7
=A�33C%�H��\)?z�@Dz�A�  C-aH                                    Bxi�5�  �          @��H��ff?���@7�A�p�C&\)��ff?�@Dz�A�\)C-�3                                    Bxi�Dj  �          @�G�����?�(�@5A�33C&����?��@B�\A��C-��                                    Bxi�S  T          @�����?�
=@5�A�33C&}q��?�@AG�A���C.                                      Bxi�a�  
�          @��\��ff?�@5A�G�C&��ff?   @A�A��\C.E                                    Bxi�p\  �          @�����  ?�  @*=qAڏ\C&  ��  ?�R@8Q�A��C,��                                    Bxi�  �          @�=q��G�?��R@'�A�Q�C&33��G�?�R@5�A�33C-�                                    Bxiύ�  
�          @�����Q�?�  @$z�A�C&��Q�?#�
@2�\A��C,                                    BxiϜN  T          @��R��
=?��R@!�A�(�C%�3��
=?#�
@/\)A噚C,��                                    BxiϪ�  
�          @��R���R?�G�@!�Aҏ\C%Ǯ���R?&ff@0  A�Q�C,}q                                    BxiϹ�  �          @������?��\@&ffAָRC%� ���?&ff@4z�A�z�C,�)                                    Bxi��@  �          @�  ��  ?�p�@#�
A�\)C&+���  ?�R@1G�A�z�C,��                                    Bxi���  T          @����?�z�@'
=Aڣ�C&�R���?
=q@333A�z�C-��                                    Bxi��  T          @���z�?��@*�HA�z�C'ff��z�>�@6ffA��HC.�)                                    Bxi��2  S          @����z�?���@*=qA�  C'�f��z�>�G�@5�A�  C.ٚ                                    Bxi��  
�          @���p�?�=q@(Q�A�Q�C'�
��p�>�@333A�z�C.��                                    Bxi�~  �          @�p���{?�=q@#33Aՙ�C'����{>�@.{A��C.}q                                    Bxi� $  �          @�z�����?�{@#�
A�C'5�����?   @/\)A���C.!H                                    Bxi�.�  
�          @�ff��{?��@(Q�A�{C'����{>�G�@333A�C.��                                    Bxi�=p  T          @����?�  @0��A��
C(z���>�33@:=qA�(�C/�f                                    Bxi�L  "          @����  ?�=q@&ffA�  C'�
��  >�@1�A��C.�                                    Bxi�Z�  T          @�  ��Q�?��@%�Aՙ�C':���Q�?�@1G�A�RC.
                                    Bxi�ib  T          @�G�����?k�@,��A��HC)������>�\)@5A�\)C0��                                    Bxi�x  
�          @�������?�ff@-p�A��HC(0�����>��@8Q�A�  C/aH                                    BxiІ�  "          @�����?�(�@%A�33C&z����?��@2�\A��
C-E                                    BxiЕT  �          @�G�����?�p�@#33A��C&W
����?(�@0��A�{C-�                                    BxiУ�  T          @�������?^�R@-p�A�p�C*0�����>k�@5�A��HC1aH                                    Bxiв�  �          @�����H?fff@0  A��
C*���H>�  @8Q�A뙚C1:�                                    Bxi��F  
�          @�33��33?p��@-p�A�ffC)�=��33>���@6ffA�33C0�f                                    Bxi���  
�          @��H��ff?��@��A���C(����ff>�ff@(Q�A��
C/
=                                    Bxi�ޒ  
�          @�������?��@�\A��HC&=q����?:�H@!G�A��HC,{                                    Bxi��8  	`          @�
=���?�=q?�A�Q�C&Y����?\(�@�A��C+�                                    Bxi���  :          @�  ��Q�?���?���A��C&@ ��Q�?^�R@p�A���C+�                                    Bxi�
�  
Z          @�����Q�?���@ ��A�{C&0���Q�?\(�@G�A�
=C+(�                                    Bxi�*  "          @����\)?�\)?�(�A��
C%�3��\)?aG�@�RA�G�C*ٚ                                    Bxi�'�  �          @�Q�����?���?�33A�G�C%�����?h��@
=qA���C*�H                                    Bxi�6v  �          @�  ����?��?�\)A�G�C%�����?k�@��A�
=C*��                                    Bxi�E  �          @�Q�����?���?�p�A��C&�{����?Tz�@�RA�{C+z�                                    Bxi�S�  "          @����
=?�@��A��C'���
=?&ff@ffA���C-B�                                    Bxi�bh  T          @�
=��?�(�@	��A�
=C'^���?0��@�A��C,Ǯ                                    Bxi�q  �          @�{��{?�33@A��RC(��{?#�
@33A�(�C-G�                                    Bxi��  
�          @��H��(�?��?�{A�ffC&z���(�?Tz�@
=A��C+:�                                    BxiюZ  
�          @��H����?��?�=qA�  C&�=����?Tz�@�A��RC+8R                                    Bxiѝ   T          @����\)?�ff?��
A�Q�C&�)��\)?\(�@�A��C+)                                    Bxiѫ�  �          @�33��p�?��
?��A��RC&����p�?W
=@�\A�\)C+=q                                    BxiѺL  	�          @�33��{?�G�?޸RA�z�C&����{?Tz�?�p�A���C+Y�                                    Bxi���  
�          @�33��?�ff?�  A���C&�=��?\(�@   A��
C+�                                    Bxi�ט  T          @����{?�G�?�\A���C&���{?Q�@G�A��HC+s3                                    Bxi��>  
�          @��
��ff?��
?�  A��HC&� ��ff?W
=@   A��C+8R                                    Bxi���  �          @�p���G�?�z�?�p�A�=qC(33��G�?8Q�?���A�Q�C,��                                    Bxi��  �          @��R��=q?�(�?�p�A�33C'����=q?J=q?�(�A�ffC+�R                                    Bxi�0  "          @�Q���(�?���?У�A|��C&��(�?h��?�33A�\)C*ٚ                                    Bxi� �  
Z          @��R��=q?��\?�Q�A��C'5���=q?W
=?�Q�A�C+u�                                    Bxi�/|  "          @�ff��=q?�G�?�
=A�33C'=q��=q?Tz�?�
=A�G�C+z�                                    Bxi�>"  
�          @������?�G�?�33A}�C'n���?W
=?�33A���C+�=                                    Bxi�L�  �          @��R���\?���?�(�A�{C'�����\?E�?���A�
=C,(�                                    Bxi�[n  �          @����ff?�
=?��
A��C'���ff?:�H@ ��A�z�C,\)                                    Bxi�j  T          @�(���
=?�33?�A�G�C(���
=?333@�A�p�C,�q                                    Bxi�x�  T          @��
��Q�?��?ٙ�A�z�C(�
��Q�?(��?�33A��C-0�                                    Bxi҇`  
�          @����Q�?�ff?ٙ�A��\C)0���Q�?�R?�33A���C-��                                    BxiҖ  T          @�����{?�\)?�{A��C(c���{?5?���A�33C,�{                                    BxiҤ�  �          @�����33?�\)?���A���C(5���33?(��@�\A�p�C-�                                    BxiҳR  "          @�����H?���?�p�A���C(
���H?!G�@(�A�\)C-L�                                    Bxi���  
�          @������\?��@ ��A�=qC(xR���\?z�@p�A�33C-�                                    Bxi�О  l          @�����H?��@�A��C(�=���H?��@�RA�=qC.#�                                    Bxi��D  �          @��\��33?���@�
A�p�C(����33?��@��A�  C.{                                    Bxi���  �          @����33?��
@ ��A��C)���33?�@��A�C.ff                                    Bxi���  �          @�G����\?�  @G�A�33C)Y����\?   @��A��\C.��                                    Bxi�6  �          @�G����?��\@�A�  C)+����>��H@��A��C.��                                    Bxi��  �          @������?�G�@�\A�\)C)33���?   @{A��HC.�H                                    Bxi�(�  �          @������?xQ�@z�A�C)�)���>�ff@\)A�z�C/�                                    Bxi�7(  �          @������?p��@z�A�C)�����>�
=@�RA��
C/z�                                    Bxi�E�  �          @�Q����?c�
@z�A�{C*p����>�p�@{A�G�C/��                                    Bxi�Tt  �          @�����=q?aG�@33A��HC*����=q>�Q�@��A�C0�                                    Bxi�c  �          @����Q�?W
=@
=qA�Q�C*�H��Q�>���@�\A�ffC0�3                                    Bxi�q�  �          @�ff��
=?Q�@��A�z�C+
��
=>�\)@�A�(�C0�                                    BxiӀf  �          @��R��\)?L��@
=qA�\)C+W
��\)>��@�A��\C10�                                    Bxiӏ  T          @�ff��
=?O\)@��A���C+5���
=>�=q@G�A�{C1�                                    Bxiӝ�  �          @�ff��\)?J=q@Q�A��C+aH��\)>��@  A�=qC1+�                                    BxiӬX  �          @�
=��  ?Tz�@�A��C*�q��  >���@��A��
C0�                                     BxiӺ�  �          @�Q���=q?B�\@�A�33C+�f��=q>k�@��A���C1}q                                    Bxi�ɤ  �          @����z�?+�@33A�p�C,޸��z�>#�
@	��A�{C2O\                                    Bxi��J  �          @�Q����\?&ff@A�=qC-���\>�@(�A�z�C2�H                                    Bxi���  �          @�\)����?0��@�
A��RC,�\����>.{@
�HA��C2#�                                    Bxi���  �          @��R����?0��@z�A�(�C,�=����>.{@
�HA�G�C2+�                                    Bxi�<  �          @�{��Q�?.{@z�A�z�C,����Q�>#�
@
�HA��C2B�                                    Bxi��  �          @�  ���\?.{@33A���C,�R���\>#�
@	��A��C2@                                     Bxi�!�  �          @����G�?5@ffA��
C,h���G�>.{@p�A��C2#�                                    Bxi�0.  T          @����=q?!G�@33A�G�C-=q��=q=�G�@��A�33C2�=                                    Bxi�>�  �          @�  ���?\)@G�A��\C.����=L��@ffA���C3xR                                    Bxi�Mz  �          @�\)����?!G�@�A��\C-@ ����=���@
�HA�ffC2�                                    Bxi�\   �          @�z���ff?.{@�A�
=C,����ff>��@�A�  C2c�                                    Bxi�j�  �          @����?+�@ffA�33C,���>�@(�A��
C2��                                    Bxi�yl  �          @�z���{?�R@�A�z�C-#���{=��
@p�A�=qC3\                                    BxiԈ  �          @�p���\)?(�@A�33C-\)��\)=��
@�A��RC3+�                                    BxiԖ�  T          @�p���
=?��@Q�A���C-z���
==L��@p�A�C3k�                                    Bxiԥ^  �          @�����ff?��@�A��RC-h���ff=u@��A��C3\)                                    BxiԴ  �          @��
��{?��@�A��
C-h���{=�\)@
=qA��C3E                                    Bxi�ª  �          @�=q��(�?
=q@�A��C.  ��(�<#�
@	��A��
C3��                                    Bxi��P  T          @�����33?�@�
A�G�C.{��33    @Q�A�G�C3�q                                    Bxi���  �          @��
���R?   @�\A�=qC.�����R��@ffA�p�C4G�                                    Bxi��  T          @��R��G�?   @ffA���C.�H��G��#�
@
=qA��C4s3                                    Bxi��B  �          @�����?   @Q�A��\C.�{��녽#�
@(�A��C4xR                                    Bxi��  �          @�
=����>��H@
=A��HC.������u@
=qA���C4��                                    Bxi��  �          @�\)��=q>��H@�
A�ffC.� ��=q�#�
@
=A�G�C4n                                    Bxi�)4  �          @�����>�@�A���C.�\��녽u@
�HA��C4��                                    Bxi�7�  �          @�Q�����>��@p�A�ffC.�3��������@��A�z�C5{                                    Bxi�F�  �          @�  ��=q>��@	��A��
C/�=��=q�\)@(�A��HC5z�                                    Bxi�U&  �          @�G����
>���@�A��
C/�����
�\)@	��A���C5�=                                    Bxi�c�  �          @�Q����\>Ǯ@��A�ffC/�\���\�#�
@
�HA���C5�R                                    Bxi�rr  �          @�
=��G�>�33@	��A���C0@ ��G��W
=@
�HA�z�C6=q                                    BxiՁ  �          @�����\>�33@�A�
=C0B����\�L��@��A��RC6�                                    BxiՏ�  �          @�Q����>��
@�A��C0������W
=@ffA��RC6G�                                    Bxi՞d  T          @�\)���\>���@z�A�p�C0p����\�L��@A��HC6.                                    Bxiխ
  T          @�����H>��
@�A�  C0�����H�aG�@ffA�33C6Q�                                    Bxiջ�  �          @�����>�{@�A�{C0c�����B�\@33A�C6                                    Bxi��V  �          @����(�>�33?�p�A�
=C0Q���(��.{@   A���C5�=                                    Bxi���  �          @�\)���H>�z�@33A��C0�f���H�u@�
A�ffC6��                                    Bxi��  �          @������>�Q�?��HA�C0!H�������?�p�A�{C5��                                    Bxi��H  �          @�����>�p�?�33A�G�C0���녾�?�
=A��C5ff                                    Bxi��  �          @�p����\>���?�\)A�(�C/�����\��Q�?�z�A���C5                                      Bxi��  �          @�p����\>\?�33A�(�C/�����\��?�
=A��C5B�                                    Bxi�":  �          @�p���=q>���?�Q�A�C0xR��=q�8Q�?���A�p�C5�                                    Bxi�0�  �          @�{���\>�Q�?���A��\C0!H���\���?�p�A��HC5�f                                    Bxi�?�  �          @�
=���
>�{?�Q�A��HC0k����
�8Q�?�(�A���C5�H                                    Bxi�N,  �          @�{���H>�z�?�Q�A��
C0޸���H�aG�?���A���C6Y�                                    Bxi�\�  �          @�  ��33>u@�A�  C1u���33���R@z�A��C7L�                                    Bxi�kx  �          @�  ���H>�  @
=A�(�C1\)���H���R@ffA�p�C7L�                                    Bxi�z  T          @�p���G�>��@ ��A��C1+���G���=q@ ��A��
C6�f                                    Bxiֈ�  �          @�p����\>�  ?�z�A�\)C1W
���\���?�z�A�G�C6                                    Bxi֗j  �          @�p����>k�?�(�A���C1�=��녾�z�?�(�A�=qC7(�                                    Bxi֦  �          @��R��33>L��@   A���C1����33���
?�p�A�=qC7u�                                    Bxiִ�  �          @��R��33>8Q�@   A�(�C2{��33��33?�p�A�(�C7�q                                    Bxi��\  �          @�������>L��@ ��A��\C1�����þ���?��RA��C7��                                    Bxi��  �          @�p���G�>L��@ ��A�(�C1ٚ��G�����?��RA���C7��                                    Bxi��  �          @����G�>B�\@   A�C1�q��G���33?�p�A��C7�q                                    Bxi��N  �          @��
���>�  @G�A�(�C1B������z�@ ��A��C7#�                                    Bxi���  �          @��\��ff>u@G�A��
C1ff��ff����@ ��A�
=C7Y�                                    Bxi��  �          @��\��p�>L��@�A�33C1Ǯ��p���Q�@�
A�G�C7�                                    Bxi�@  �          @��
��ff>8Q�@ffA���C2  ��ff�\@z�A�G�C8+�                                    Bxi�)�  �          @�=q��z�>#�
@��A���C2:���z���@
=A���C8�
                                    Bxi�8�  T          @��H��ff>#�
@�\A��C2J=��ff�Ǯ@ ��A��HC8O\                                    Bxi�G2  �          @�p����>�?�p�A��
C2�)��녾���?���A���C8T{                                    Bxi�U�  �          @�������=�G�@�A�=qC2�\���׾�(�?�p�A�z�C8�3                                    Bxi�d~  �          @����\)=�G�@�A�C2���\)��(�?��RA�  C8��                                    Bxi�s$  �          @�z�����=�G�?��RA��C2�����þ�(�?���A�C8�H                                    Bxiׁ�  �          @�����=�Q�@   A�\)C2�R�����ff?��HA��C8�)                                    Bxiאp  �          @��
��  =��
@   A��RC3&f��  ��?���A�=qC9�                                    Bxiן  �          @�33���R=��
@33A��
C3����R���@   A��C9+�                                    Bxi׭�  �          @�(�����=#�
?�p�A���C3�����׿   ?�A�p�C9h�                                    Bxi׼b  �          @��
���<�@ ��A��C3�3�����\?�Q�A�C9��                                    Bxi��  �          @����Q�=L��?��HA��C3}q��Q��?�33A��RC9E                                    Bxi�ٮ  
�          @��H���<��
?��HA�=qC3� �����\?��A�ffC9�=                                    Bxi��T  T          @����Q�<��
?��HA�C3Ǯ��Q��\?�33A��
C9�\                                    Bxi���  �          @�33��Q�<#�
?�33A��RC3���Q��\?�=qA��RC9��                                    Bxi��  �          @����G�    ?�\)A�{C4
=��G���\?�ffA��C9��                                    Bxi�F  �          @����G�    ?��A���C4���G���?�A���C9��                                    Bxi�"�  �          @�z����\��?�{A�=qC4J=���\��?��A���C9��                                    Bxi�1�  �          @�z����<#�
?�A��RC3޸��녿�\?���A��RC9��                                    Bxi�@8  T          @�(���=q���
?���A��RC4:���=q��?޸RA�Q�C9��                                    Bxi�N�  �          @�33�����u?�ffA�Q�C4�)�������?�(�A�
=C9��                                    Bxi�]�  �          @�����\)���
?��A�33C4���\)��?ٙ�A�33C:@                                     Bxi�l*  �          @�=q���þ�?��
A�33C5k����ÿ�R?�
=A�{C:��                                    Bxi�z�  �          @�p����
�#�
?��A�33C5� ���
�&ff?�A�p�C:��                                    Bxi؉v  �          @����z�\)?�(�A�\)C5�=��z�(�?�{A�Q�C:}q                                    Bxiؘ  
�          @�z���z�B�\?�33A��C6  ��z�#�
?��
A|  C:�q                                    Bxiئ�  
�          @�����
���?���A�=qC6�����
�333?��HAqG�C;c�                                    Bxiصh  �          @�33�����L��?�A���C4� �������?�(�A���C9��                                    Bxi��  �          @�z�������?�A�\)C4G������\)?�A�(�C:                                    Bxi�Ҵ  �          @��R��(����?�z�A��RC5�f��(��.{?�ffA���C;:�                                    Bxi��Z  �          @�p����H��?��A�C5L����H�#�
?��
A�ffC:޸                                    Bxi��   �          @�p���(��8Q�?��
A��\C5���(��+�?�z�A�=qC;!H                                    Bxi���  �          @������
�B�\?޸RA��
C6���
�+�?�\)A�\)C;�                                    Bxi�L  �          @�(�����u?�Q�A�(�C6������333?ǮA��RC;s3                                    Bxi��  �          @�z�����u?�p�A�33C6������8Q�?���A�p�C;��                                    Bxi�*�  �          @��H��녾W
=?޸RA���C6G���녿0��?�{A���C;n                                    Bxi�9>  �          @����33�k�?�
=A���C6u���33�333?�ffA�(�C;h�                                    Bxi�G�  �          @��
����W
=?�
=A��HC6J=����.{?�ffA�
C;:�                                    Bxi�V�  �          @�(����
���
?��A�C7^����
�E�?��RAt��C<&f                                    Bxi�e0  �          @�����zᾔz�?�33A��
C7\��z�=p�?�  AvffC;�)                                    Bxi�s�  �          @��
��33��=q?��HA�\)C6ٚ��33�=p�?ǮA��HC;�H                                    Bxiق|  �          @�����zᾊ=q?�z�A��C6����z�:�H?\AyC;��                                    Bxiّ"  �          @�(���33���?�(�A���C6�=��33�=p�?�=qA�ffC;�f                                    Bxiٟ�  �          @�����
���?޸RA��C6����
�@  ?���A�33C;�                                    Bxiٮn  �          @�{�����z�?�G�A�C7�����G�?���A���C<0�                                    Bxiٽ  �          @�ff���;���?��A�ffC7�����ͿTz�?�\)A�  C<�=                                    Bxi�˺  �          @���(����R?�\A���C7E��(��O\)?�{A�C<��                                    Bxi��`  �          @������
���
?޸RA���C7c����
�O\)?�=qA��C<��                                    Bxi��  �          @��
��33���R?��HA�p�C7\)��33�J=q?��A
=C<p�                                    Bxi���  �          @�p���zᾅ�?�(�A�G�C6�\��z�@  ?���A��\C;�                                    Bxi�R  T          @�{��p���z�?��HA��C7��p��E�?�ffA}�C<�                                    Bxi��  �          @�ff��p�����?�p�A��C7���p��J=q?���A�  C<@                                     Bxi�#�  �          @��R��{��z�?�
=A�G�C7���{�E�?��
AxQ�C<                                    Bxi�2D  �          @�p����;�{?�
=A��C7�����ͿO\)?�G�Aw
=C<��                                    Bxi�@�  �          @�����(�����?�A���C833��(��^�R?�p�As\)C=#�                                    Bxi�O�  �          @������
���?�
=A���C8c����
�c�
?��RAt��C=^�                                    Bxi�^6  �          @�����(���(�?�A��C8����(��fff?�(�Aqp�C=�=                                    Bxi�l�  �          @���z��(�?�(�A���C8����z�k�?\Aw�
C=�f                                    Bxi�{�  �          @�ff�����?ٙ�A���C8O\���c�
?�G�At��C=Q�                                    Bxiڊ(  �          @�ff��{��p�?�A�Q�C7�H��{�Y��?��RAr{C<�
                                    Bxiژ�  �          @���p���p�?�
=A���C7�H��p��Y��?�  AtQ�C<��                                    Bxiڧt  �          @����zᾣ�
?�Q�A��C7s3��z�O\)?\Ay�C<�\                                    Bxiڶ  
�          @��
������R?�Q�A��
C7B�����J=q?\A{\)C<k�                                    Bxi���  �          @�(����
��z�?�z�A���C7\���
�E�?�  Ax  C<&f                                    Bxi��f  �          @�(���(��\?���A�Q�C8���(��W
=?�Ai�C<�f                                    Bxi��  �          @������?��RAt��C:
=���}p�?�  AL��C>^�                                    Bxi��  �          @��������H?ǮA��C98R����n{?��A]p�C=޸                                    Bxi��X  �          @�ff��
=�
=q?�p�A|��C9����
=�u?�G�AU�C>h�                                    Bxi��  �          @����  ��?\A�Q�C9�)��  �s33?��AY�C>8R                                    Bxi��  �          @�������?ǮA�z�C9.����k�?���Ac�C>                                      Bxi�+J  �          @�\)��
=��?˅A��HC95���
=�n{?�\)Ah  C>!H                                    Bxi�9�  �          @��R���R��ff?���A�{C8�����R�fff?�\)Ag�
C=��                                    Bxi�H�  �          @�
=���R���H?˅A�33C9Q����R�p��?�\)Ah  C>B�                                    Bxi�W<  �          @�ff��{��?�=qA�G�C9���{�k�?�\)AiG�C>                                    Bxi�e�  �          @�  ��Q�   ?��A�=qC9xR��Q�s33?���A]p�C>8R                                    Bxi�t�  �          @������\)?��A���C:\�����  ?�ffA[�C>�\                                    Bxiۃ.  �          @�p���p���?\A�(�C9��p��u?��A[�
C>�                                     Bxiۑ�  �          @�\)��
=���?���A���C9�R��
=��  ?�=qA`  C>��                                    Bxi۠z  �          @�
=��
=��\?ǮA���C9�
��
=�u?�=qAap�C>xR                                    Bxiۯ   �          @�
=��
=��\?ǮA��RC9�H��
=�xQ�?�=qA`��C>�                                    Bxi۽�  �          @�ff��{���?ǮA���C:
=��{��  ?��A^�RC>��                                    Bxi��l  �          @����\)?��
A�33C:+�����  ?��A[33C>��                                    Bxi��  �          @�p�����\)?���A���C:.������\?���AaC?#�                                    Bxi��  �          @�z���(��z�?�ffA��C:p���(����
?��A^{C?W
                                    Bxi��^  �          @��
����z�?��A�
=C:� ������
?��
A\��C?c�                                    Bxi�  �          @�ff���
=?���A�(�C:s3�����?��A_
=C?c�                                    Bxi��  �          @��R�����?�{A��C:�
������?��Ac�
C?��                                    Bxi�$P  �          @��R��{���?ǮA���C:����{���?��A[
=C?��                                    Bxi�2�  T          @�\)��
=��R?��A�(�C:��
=����?��\AU��C?��                                    Bxi�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�PB              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�m�   f          @�  ��\)�.{?ǮA��
C;p���\)���?��\AT��C@G�                                   Bxi�|4  �          @�\)���R�#�
?ǮA��\C;  ���R����?��
AXz�C?�                                   Bxi܊�  �          @��R��ff�&ff?\A�p�C;���ff���?��RAR{C?�)                                   Bxiܙ�  �          @�\)���R�+�?�ffA�C;^����R����?�G�AT��C@@                                    Bxiܨ&  �          @�Q���Q�0��?�  A|��C;k���Q쿐��?��HAJ{C@�                                   Bxiܶ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi��r   f          @����
=�B�\?�G�A~�HC<Q���
=����?�Q�AG�C@�R                                    Bxi��  �          @�Q���  �@  ?\A33C<{��  ��Q�?���AH��C@�                                    Bxi��  T          @������׿L��?��A��HC<�����׿�  ?���AHz�CAY�                                    Bxi��d  �          @������ÿJ=q?\A33C<xR���ÿ�p�?�Q�AF�RCA#�                                    Bxi� 
             @�Q���  �O\)?��HAu�C<�R��  ��p�?�\)A;�
CA.                                   Bxi��  �          @����G��G�?\A}�C<W
��G���(�?�Q�AE�CA                                      Bxi�V  �          @�G���Q�@  ?�ffA��C<!H��Q쿚�H?�(�AL  C@�                                    Bxi�+�  �          @�����  �8Q�?�=qA���C;�
��  ��Q�?�G�AR�HC@��                                    Bxi�:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�IH              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�W�  
*          @�����G��B�\?\A}�C<+���G����H?�Q�AF{C@��                                   Bxi�f�  �          @����G��@  ?�G�A|��C<���G����H?�Q�AE�C@�{                                   Bxi�u:  �          @�����Q�J=q?�p�Ax(�C<���Q쿜(�?��A>�\CA)                                   Bxi݃�  �          @�������O\)?��
A���C<�{������\?�
=AF=qCA�)                                   Bxiݒ�  �          @�����Q�J=q?�p�Ax��C<� ��Q쿝p�?��A?33CA!H                                   Bxiݡ,  �          @�\)���R�O\)?��RA|Q�C<ٚ���R��  ?��A@z�CA��                                   Bxiݯ�  T          @����p��L��?�\)Aj�RC<�{��p����H?��
A/�
CA&f                                   Bxiݾx  �          @�����{�E�?�G�AW\)C<ff��{���?n{A�C@Y�                                    Bxi��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi���  	�          @����ff�L��?�p�AR�RC<�R��ff��z�?fffA�C@��                                   Bxi��j  �          @�p����׿B�\?�  A*ffC<@ ���׿�ff?0��@�33C?J=                                    Bxi��  �          @�{�����E�?xQ�A$  C<B�������ff?(��@�ffC?+�                                    Bxi��  �          @�ff��녿:�H?}p�A'�C;�=��녿�G�?0��@��C>�\                                    Bxi�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�%   f          @���G��@  ?uA"�\C<\��G����
?&ff@��C>�R                                    Bxi�3�  �          @��R��=q�=p�?}p�A&�RC;����=q���
?.{@�C>��                                    Bxi�BN  �          @�{��녿&ff?�G�A*�HC:����녿s33?:�H@�  C>#�                                    Bxi�P�  �          @�  ����.{?�G�A)p�C;=q����z�H?8Q�@�=qC>\)                                    Bxi�_�  T          @�ff��녿0��?�  A)G�C;\)��녿z�H?5@�Q�C>xR                                    Bxi�n@  T          @�����  �:�H?�  A*=qC;���  ���\?0��@�(�C?                                    Bxi�|�  �          @�(���
=�@  ?��A5��C<8R��
=����?=p�@�{C?��                                    Bxiދ�  �          @�(�����E�?z�HA(Q�C<W
�����ff?+�@�C?c�                                    Bxiޚ2  �          @����ff�B�\?�ffA4z�C<O\��ff����?:�H@��\C?��                                    Bxiި�  �          @��
���R�E�?�  A,(�C<s3���R����?.{@���C?��                                    Bxi޷~  �          @�33���R�L��?p��A!�C<�����R����?(�@�G�C?�)                                    Bxi��$  �          @�z�����J=q?xQ�A%G�C<�{�������?#�
@�=qC?�\                                    Bxi���  �          @�����R�E�?}p�A*{C<n���R���?+�@�z�C?��                                    Bxi��p  �          @����
=�@  ?xQ�A&=qC<.��
=���
?&ff@߮C?8R                                    Bxi��  �          @��H��{�G�?uA%��C<�{��{���?!G�@��C?�{                                    Bxi� �  �          @�=q����L��?z�HA*�\C<�=������?&ff@��C?��                                    Bxi�b  �          @�33��{�O\)?z�HA)p�C<�f��{����?#�
@��C?�R                                    Bxi�  T          @�=q����Q�?}p�A,��C=������{?&ff@�G�C@0�                                    Bxi�,�  T          @��\��p��Tz�?�G�A.�HC=
��p���\)?(��@��C@G�                                    Bxi�;T  �          @��\��p��Q�?�  A-C<�R��p���{?(��@�(�C@#�                                    Bxi�I�  �          @�33���O\)?�=qA;33C<��������?=p�@�C@^�                                    Bxi�X�  �          @�=q���ͿJ=q?��A8��C<� ���Ϳ�{?8Q�@�=qC@.                                    Bxi�gF  �          @��
���Q�?�z�AG�C<������?L��A	C@��                                    Bxi�u�  �          @�����=p�?���AC33C<.�����?L��A	C?�f                                    Bxi߄�  �          @��
��{�Y��?�=qA:{C=B���{��?8Q�@�ffC@��                                    Bxiߓ8  �          @�33����W
=?�{A?�C=8R�����?@  A z�C@�=                                    Bxiߡ�  �          @�G���33�aG�?��A>�\C=�q��33����?8Q�@�G�CAB�                                    Bxi߰�  �          @����(��O\)?�{AA��C<�q��(���33?B�\A\)C@��                                    Bxi߿*  �          @��\����G�?�{A@  C<�������\)?B�\A�
C@@                                     Bxi���  �          @�33���O\)?��A;33C<�������?:�H@�33C@h�                                    Bxi��v  �          @����z�W
=?��A8��C=E��z῔z�?333@�\C@�R                                    Bxi��  
�          @��\���ͿW
=?�=qA;
=C=:����Ϳ�?8Q�@��RC@�q                                    Bxi���  �          @����(��Tz�?��A8��C=0���(���33?333@��HC@��                                    Bxi�h  �          @��H���J=q?��
A1G�C<��������?.{@��C@�                                    Bxi�  �          @�Q���33�L��?�ffA7�
C<���33��\)?333@�33C@c�                                    Bxi�%�  �          @������
�E�?��A6�\C<�{���
���?333@�z�C@�                                    Bxi�4Z  �          @�G���(��:�H?�ffA6�HC<#���(����?8Q�@���C?��                                    Bxi�C   �          @����z�=p�?��A=��C<5���zῊ=q?@  A�\C?��                                    Bxi�Q�  �          @�=q��z�aG�?��A7�C=����zῙ��?.{@�G�CA�                                    Bxi�`L  �          @�=q���ͿJ=q?��A8  C<�����Ϳ�\)?5@�z�C@@                                     Bxi�n�  �          @��H���8Q�?�=qA9�C;�������?=p�A (�C?�
                                    Bxi�}�  �          @�33���G�?��A;�C<������\)?:�H@�z�C@+�                                    Bxi��>  �          @����p��0��?�  A-C;�)��p���  ?.{@��C>�q                                    Bxi���  �          @�=q��p��=p�?}p�A,  C<(���p���ff?(��@�C?xR                                    Bxi੊  �          @��\��ff�8Q�?xQ�A&�RC;���ff���\?#�
@��
C?#�                                    Bxi�0  �          @��R�����O\)?�  A)p�C<�R������\)?#�
@׮C?�                                    Bxi���  �          @�p���  �Y��?��
A0  C=+���  ��z�?(��@�ffC@��                                    Bxi��|  �          @����{�aG�?��A3�C=�H��{����?&ff@߮CA                                    Bxi��"  �          @�����u?}p�A*=qC>s3����  ?�@�z�CA��                                    Bxi���  �          @�����ff����?}p�A(Q�C?�
��ff����?�@��
CB�{                                    Bxi�n  �          @�����H��
=?p��A#�CA)���H����>�G�@���CC�H                                    Bxi�  �          @����zΐ33?�  A,Q�C@����zῷ
=?�\@��CC�{                                    Bxi��  �          @�����׿��?�=qA6ffC?�q���׿�?(�@˅CC�                                    Bxi�-`  �          @�����Ϳ��?��
A2{C@  ���Ϳ�33?\)@�  CC0�                                    Bxi�<  T          @�����
���\?z�HA)��C?J=���
���?�@�  CBY�                                    Bxi�J�  �          @�����Ϳ���?��A5�C?���Ϳ��?
=@�G�CC\                                    Bxi�YR  �          @����(���{?���A>{C@=q��(���Q�?�R@��HCC�3                                    Bxi�g�  �          @�33�����33?�{A@Q�C@�3�����p�?(�@�G�CD.                                    Bxi�v�  �          @�z������=q?���A<z�C?�)�����z�?�R@�33CCT{                                    Bxi�D  �          @����R���?�{A<��C?�����R��?!G�@��
CCL�                                    Bxi��  �          @�������=q?��A9C?� ����33?(�@�
=CC(�                                    Bxiᢐ  �          @��������?���A<��C?������33?!G�@�CC�                                    Bxi�6  �          @�33��z῏\)?uA&{C@W
��zῳ33>��@��HCCE                                    Bxi��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�΂   e          @�(���ff��z�?J=qA�C@�)��ff����>���@L��CB��                                    Bxi��(  �          @�(���p���Q�?aG�A�\C@���p���
=>�p�@|(�CCxR                                    Bxi���  �          @�z�����Q�?fffA{C@�����Q�>Ǯ@�z�CC��                                    Bxi��t  
�          @��
�����
=?aG�A�RC@�{�����>�p�@~{CCaH                                    Bxi�	  �          @�(������p�?fffAp�CAs3�����p�>�p�@x��CD                                    Bxi��  �          @�(���zῢ�\?k�A=qCA����z���
>�p�@\)CD�=                                    Bxi�&f  �          @��
���
��ff?p��A!p�CB8R���
�Ǯ>\@���CD��                                    Bxi�5  �          @�z����Ϳ��H?��
A0  CA8R���Ϳ�G�>��H@�  CDW
                                    Bxi�C�  �          @������Ϳ�p�?��A5CAh����Ϳ��?�@�  CD��                                    Bxi�RX  �          @������
��G�?�AH  CA�{���
��{?��@�z�CEu�                                    Bxi�`�  �          @�{��33����?��A\(�CC���33��G�?+�@�33CG
                                    Bxi�o�  �          @�����녿��?��\AZ{CCaH��녿�\?&ff@ۅCGO\                                    Bxi�~J  �          @��
��Q쿫�?�33AqG�CB�3��Q��\?G�A��CGu�                                    Bxi��  �          @����ff��p�?�
=Aw�CD����ff��?B�\Ap�CI#�                                    Bxi⛖  �          @����\)��(�?�=qAfffCDs3��\)��\)?+�@���CH�{                                    Bxi�<  �          @�����R���?�=qAe��CE8R���R��
=?#�
@��CIB�                                    Bxi��  T          @�����Q��  ?���Ag\)CD����Q��33?+�@�CH�                                    Bxi�ǈ  �          @������׿�
=?��An�RCC�����׿���?:�H@���CHJ=                                    Bxi��.  �          @�z���=q��p�?��AeCA�f��=q��33?B�\AG�CF
=                                    Bxi���  �          @��\��
=����?��Ad(�CC����
=���
?+�@���CG��                                    Bxi��z  �          @����  ����?�{Ak�CCk���  ��ff?8Q�@�  CG��                                    Bxi�   �          @�ff���
��{?��
AX��CB޸���
��  ?&ff@�33CF�)                                    Bxi��  �          @�{���
����?��\AXQ�CBs3���
���H?(��@޸RCFu�                                    Bxi�l  �          @��H��\)����?��Ah��CC!H��\)��G�?5@���CG}q                                    Bxi�.  �          @�����z῵?��At��CD8R��z����?:�H@�p�CH                                    Bxi�<�  �          @�z����H���?���AM�CBp����H��?
=@ȣ�CF33                                    Bxi�K^  �          @�z����\���?�p�AS�CB}q���\��Q�?�R@��
CFh�                                    Bxi�Z  �          @��H���ÿ��\?��
A]G�CB#����ÿ�?.{@�G�CFT{                                    Bxi�h�  �          @��
��녿�  ?��A_33CA����녿�z�?333@�\)CF(�                                    Bxi�wP  
�          @����녿��H?��
A]CA}q��녿�\)?333@�G�CE�                                    Bxi��  �          @�33�������?���Ai��CC.������
?333@�=qCG�{                                    Bxi㔜  �          @�=q��
=��=q?���Ak\)CC��
=��G�?5@�\)CG}q                                    Bxi�B  �          @��\��  ��ff?��Ac33CB����  ���H?0��@��CF�H                                    Bxi��  �          @�G����R��p�?��As\)CA�����R��
=?J=qA	�CF��                                    Bxi���  �          @�������
=?�z�Av{CAT{������?Tz�A
=CF=q                                   Bxi��4  	-          @����G���
=?��Apz�CA.��G����?O\)A
ffCE�R                                   Bxi���  U          @�33��Q쿠  ?�{Ak�
CB���Q��Q�?@  A�CF�)                                   Bxi��  �          @�33��
=��G�?�  A��RCB+���
=��  ?aG�A\)CGc�                                   Bxi��&  �          @�33���׿�
=?�33AF=qCC����׿�\>�@�z�CGp�                                   Bxi�	�  �          @�  ���Ϳ˅?uA(z�CE�R���Ϳ�>u@)��CH�{                                   Bxi�r  �          @�������(�?p��A%CG�=������H>.{?��CI�                                   Bxi�'  �          @�
=��=q��G�?s33A'�CH���=q���R>#�
?��
CJ�                                    Bxi�5�  
�          @����p���{?�33AK�CC� ��p����H?�\@�=qCG33                                   Bxi�Dd  �          @�\)��ff��
=?�Q�AQCAaH��ff�Ǯ?(�@�
=CEp�                                   Bxi�S
  �          @�{��zΰ�
?�Q�ATQ�CB�f��z��33?z�@��
CF�f                                   Bxi�a�  �          @�p����
���?��ABffCCW
���
��z�>�@��CF�
                                   Bxi�pV  �          @������R��Q�?���A@(�CD+����R��G�>�
=@�33CG��                                   Bxi�~�  �          @����\)����?���AEG�CD5���\)���
>�ff@��
CG�3                                   Bxi䍢  �          @�33��Q쿼(�?��ADQ�CDc���Q��>�G�@�\)CG�
                                   Bxi�H  �          @��\��  ��(�?���A>�RCDn��  ���>��@�(�CG��                                   Bxi��  �          @������R����?�z�AK\)CC�\���R�޸R?   @�CGG�                                   Bxi乔  �          @�Q�������R?�\)ADz�CB����˅?�\@��CE��                                   Bxi��:  �          @����녿˅?�  A]p�CFG���녿��H>��H@���CJ+�                                   Bxi���  �          @�ff������?�ffAh��CJ\������>�
=@��
CM�{                                   Bxi��  
�          @�ff��\)��G�?�  A_33CH^���\)��>�
=@�CL�                                   Bxi��,  
�          @�{��33��?�{As�CA�\��33��\)?B�\A33CF��                                   Bxi��  �          @������z�?���Ai�CA5�������?:�HA (�CE��                                    Bxi�x  �          @�\)��p�����?���AT  CC��p��ٙ�?��@�G�CG�                                    Bxi�   �          @��R��(���?���AH��CD5���(���G�>�ff@�p�CG�\                                    Bxi�.�  �          @��R����Ǯ?�G�A3�CE� �����>�\)@ECH�                                    Bxi�=j  �          @��\��{�У�?�{A@��CF@ ��{��Q�>�{@h��CIs3                                    Bxi�L  �          @��H���R��
=?�ffA5�CF�����R���H>��@1�CI�)                                    Bxi�Z�  �          @�Q����
��ff?W
=A  CHG����
���R<��
>��CJ5�                                    Bxi�i\  �          @�  ��=q����?��\A2ffCH����=q��>.{?�CKT{                                    Bxi�x  �          @�G���33��=q?��
A4Q�CH����33�>8Q�?�(�CKL�                                    Bxi冨  �          @������H���?��A9CHT{���H�z�>aG�@�CK+�                                    Bxi�N  �          @��������?�=qA<Q�CH�R�����>aG�@ffCK�
                                    Bxi��  �          @����(���{?n{A!�CH�)��(���=�\)?B�\CK{                                    Bxi岚  
�          @������H��z�?}p�A,��CI�
���H�	��=���?��CL                                      Bxi��@  �          @�33��(���p�?�\)AmG�CG���(����?�@��\CK��                                    Bxi���  �          @�������  ?�Q�AyG�CGǮ����(�?z�@��CLB�                                    Bxi�ތ  �          @�z���z��(�?�(�A}G�CGff��z���?(�@�Q�CL�                                    Bxi��2  �          @�����=q��=q?�(�A�
=CF5���=q�33?(��@�\)CK�                                    Bxi���  �          @������\��
=?�z�Av=qCG33���\��?�@���CK�                                    Bxi�
~  �          @�33����ٙ�?���A|(�CGB�����	��?��@�ffCK��                                    Bxi�$  �          @����녿ٙ�?�(�A�(�CGn����
=q?(�@ӅCL#�                                    Bxi�'�  �          @�\)���׿�=q?���A��\CF\)�����33?#�
@ᙚCK8R                                    Bxi�6p  �          @�Q����\��ff?�33Aw�
CE� ���\���R?�R@�  CJs3                                    Bxi�E  �          @�ff��
=����?�G�A���CFp���
=��
?333@�\)CK�H                                    Bxi�S�  �          @�
=��  ���
?\A��
CEٚ��  ��?:�HA ��CK(�                                    Bxi�bb  T          @����
=��Q�?��RA���CD����
=��Q�?=p�A�CJT{                                    Bxi�q  �          @�  ���\��p�?�
=A|Q�CE����\��Q�?(��@�\)CI�                                    Bxi��  �          @�Q���=q���R?�p�A�z�CE8R��=q��p�?333@�(�CJW
                                    Bxi�T  �          @������H���H?���A�Q�CD����H���R?L��A
�HCJ\)                                    Bxi��  �          @�G����\��Q�?�{A�\)CD�{���\���R?W
=Ap�CJ\)                                    Bxi櫠  �          @�����G�����?�33A�=qCD����G�� ��?^�RA��CJ�\                                    Bxi�F  �          @�{��ff���R?��HA�=qCD����ff�z�?fffA�CJ�q                                    Bxi���  �          @�z���p����H?��A�\)CD}q��p��G�?Y��AG�CJL�                                    Bxi�ג  �          @�������(�?�z�A��HCD�
����\?\(�A33CJxR                                    Bxi��8  �          @���{��33?�G�A�G�CCٚ��{�G�?z�HA'
=CJE                                    Bxi���  �          @�{��ff���H?ٙ�A�(�CDn��ff�33?h��ACJ�                                    Bxi��  T          @��
��p����R?�ffA��CD����p�� ��?@  A Q�CJG�                                    Bxi�*  T          @�z���p����R?���A�=qCD�{��p���?L��AQ�CJu�                                    Bxi� �  �          @�{��  ��(�?���A�ffCDff��  � ��?G�A  CI�H                                    Bxi�/v  �          @�z���
=��?��
A�CC�R��
=��Q�?B�\A��CI\)                                    Bxi�>  �          @����ff��z�?�p�A�=qCC�)��ff��z�?8Q�@��CI
                                    Bxi�L�  �          @�=q��p�����?˅A���CBT{��p����?\(�A�CG�q                                    Bxi�[h  �          @��\��ff��?��A��C@����ff��\?xQ�A z�CF                                    Bxi�j  �          @�=q��{����?У�A�  CA���{���?p��A�CG\                                    Bxi�x�  �          @�=q�����H?��A�\)CA)����ff?s33A��CG0�                                    Bxi�Z  �          @������H���?У�A��CCL����H���H?Y��AffCI�                                    Bxi�   �          @��\��{����?��A�ffCB�H��{���?J=qA�HCH
=                                    Bxi礦  �          @��H���Ϳ�{?�\)A���CB�q���Ϳ�
=?\(�ACH�                                     Bxi�L  �          @�(�������?��A�33CC(�����z�?��A3
=CJ
                                    Bxi���  �          @�33��33����?�33A��RCBxR��33�G�?���A:{CI�{                                    Bxi�И  �          @�33��z��=q?�(�A�  CE�H��z��z�?!G�@�  CJ�f                                    Bxi��>  �          @����R���?ٙ�A�
=CC�����R����?n{A%�CJ��                                    Bxi���  �          @�Q���
=��
=?���A���CB�q��
=��z�?�p�Ab�\CK.                                    Bxi���  �          @�=q���H��=q?��
A�p�CA�q���H��  ?�\)AV�HCJ                                    Bxi�0  �          @�33��33��ff?�{A�=qCA���33��G�?���AeCJ                                    Bxi��  �          @�=q���\����?���A��CAٚ���\��\?�z�A_33CJ+�                                    Bxi�(|  �          @����Q쿜(�?�A�Q�CD���Q��p�?:�HA��CJ{                                    Bxi�7"  �          @�  �\)��?�(�A�CDL��\)��?��\AR{CLc�                                    Bxi�E�  �          @�33�~�R�}p�@A�ffCB�~�R����?�
=A��CL��                                    Bxi�Tn  �          @�{��G����@Q�A�  CB�R��G���33?�
=A�ffCM(�                                    Bxi�c  �          @�\)�x�ÿ�
=?��Aƣ�CD�)�x�ÿ��?�z�Ao�CM��                                    Bxi�q�  �          @���tz�xQ�?��HA�p�CB0��tz�޸R?���A��
CLxR                                    Bxi�`  T          @�=q�u���G�?�p�A��CL��u��ff?B�\A��CSz�                                    Bxi�  
�          @�(��z�H��  @ffA�z�CE���z�H��
?���A��RCO��                                    Bxi蝬  �          @���z=q����@�A�CC�H�z=q����?��HA�G�CN�=                                    Bxi�R  �          @�\)�tzῈ��@z�A�(�CC�{�tz���?�\)A���CN0�                                    Bxi��  �          @��
�o\)�u@�\A�=qCBk��o\)��\?��A�{CM\)                                    Bxi�ɞ  �          @�(��u�W
=?�Q�A��
C@Q��u��\)?���A�CJ�H                                    Bxi��D  �          @�����  �k�?��HA�ffC@����  �ٙ�?��A���CK�                                    Bxi���  �          @��R��33���@�
A��CB0���33����?�\)A�CLY�                                    Bxi���  �          @��R�|�Ϳ�=q@G�A�Q�CCQ��|�Ϳ��R?��A���CN                                    Bxi�6  �          @�p��������@p�A�z�CC
=����ff?ٙ�A���CO
=                                    Bxi��  �          @�{�����333@�A�
=C=B�������
=?�G�A���CIT{                                    Bxi�!�  �          @��R��=q�Q�@�A�33C>0���=q���?��HA�G�CI^�                                    Bxi�0(  T          @��R��녿J=q@�
Aޏ\C>s3��녿�G�?ٙ�A��\CJ0�                                    Bxi�>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�Mt              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�\   2          @�G��y���h��@(Q�B�CA+��y��� ��?�A��COB�                                   Bxi�j�  �          @���x�ÿY��@$z�B=qC@aH�x�ÿ�
=?�33A��RCNc�                                   Bxi�yf  
�          @�
=�tz�E�@*=qB	G�C?^��tz��33@G�A�ffCNk�                                   Bxi�  �          @��R�u��+�@*�HB	��C=�)�u���@z�A�(�CML�                                   Bxi閲  �          @�{�u��z�@(Q�B��C<�
�u���(�@�Aҏ\CL+�                                   Bxi�X  �          @�G��u��  @�HB �C7Ǯ�u��=q@�\A�33CG0�                                    Bxi��  �          @�G��w
=�W
=@��A�(�C7��w
=���
@�A�ffCF^�                                    Bxi�¤  �          @�ff���Y��@<��B
�C?k����
=@\)A�  CN�R                                    Bxi��J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi��  	�          @�{��G���R@5B��C<+���G���{@\)Ạ�CKaH                                   Bxi��<  �          @�G���Q�G�@%�A�z�C>c���Q���?�
=A�=qCK�)                                   Bxi��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�).  
�          @�33���R<�@7�B	33C3�f���R���@#�
A�
=CE:�                                   Bxi�7�  T          @����=q=�G�@>{B
=qC2���=q���
@,(�A��RCD�                                     Bxi�Fz  T          @�ff��G�=�\)@;�B	Q�C3)��G���ff@(Q�A���CDٚ                                    Bxi�U   �          @��
����=L��@1G�B{C3G��������R@�RA�CD�                                    Bxi�c�  
�          @����z��@#33A��RC4n��zῚ�H@\)AۅCDL�                                    Bxi�rl  �          @�\)��p�>B�\@.{B�\C1p���p�����@   A�RCB�\                                    Bxi�  �          @����p�>�33@5�Bz�C/B���p���G�@*�HA��CA�                                    Bxiꏸ  T          @�ff�~�R?z�@8Q�BG�C+���~�R�Tz�@4z�B
�HC?�R                                    Bxi�^  �          @�{�u�?\(�@@  B�\C'Y��u��#�
@C33BC=z�                                    Bxi�  
�          @��\�w
=?p��@J=qB(�C&W
�w
=�&ff@N�RBQ�C=�=                                    Bxi껪  
�          @��\�n{?�@P  B33C"�)�n{���H@Z�HB)=qC;�                                     Bxi��P  �          @��H���
����@�RA�RC9� ���
��33?��A���CF�R                                    Bxi���  �          @��������@z�A�
=C9p�������=q?��A�  CE8R                                    Bxi��  �          @�G����
?(��@3�
B�
C*ٚ���
�:�H@2�\B�C>{                                    Bxi��B  �          @�Q���33=���@p�A�  C2����33����@{A�
=CA�                                    Bxi��  �          @�����Q쾏\)@
=qA���C7�)��Q쿢�\?��
A�=qCC�R                                    Bxi��  �          @�����
��R?�
=A��C;�����
����?�z�A��
CEn                                    Bxi�"4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�0�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�?�  
�          @��H��G����@z�AθRC5�q��G���(�?�p�A�ffCBT{                                    Bxi�N&  �          @�33��=q�s33?���An�HC>�\��=q��p�?E�A�
CDL�                                    Bxi�\�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxi�kr  &          @�����G���
=?z�@˅CC�{��G�����#�
��\CD�                                    Bxi�z  T          @�G���G���
=?#�
@��CC�=��G��Ǯ�����\CE(�                                    Bxi눾  �          @���zῙ��?�p�AP��CA#���z���?�\@�(�CE��                                    Bxi�d  �          @�z����
��ff?��A5�CB0����
��33>��
@Y��CE��                                    Bxi�
  �          @������ÿ�ff?��
A4(�CB�����ÿ��>�z�@H��CF�                                    Bxi봰  �          @�33���\���?���A9G�CBJ=���\��33>���@aG�CE��                                    Bxi��V  �          @��
��33��ff?�ffA4z�CBO\��33���>���@N�RCE޸                                    Bxi���  �          @��H��녿���?��HAR�\CAL���녿У�?   @�33CE޸                                    Bxi��  �          @��\��������?�Q�AMCAT{������\)>��@�=qCE�                                     Bxi��H  �          @�(��������?�G�AX��C@0��������?z�@�CE�                                    Bxi���  �          @��������{?��AC33C@#�����\>��@�Q�CDk�                                    Bxi��  �          @�p���zῥ�?�\)A?�CB���z��>�p�@z=qCF                                    Bxi�:  �          @�p����
����?�
=AICBp����
��p�>��@���CF�)                                    Bxi�)�  
�          @�z���녿�G�?���A<Q�CD�
��녿���>u@$z�CH{                                    Bxi�8�  T          @�(����ÿ�G�?�33AECD�����ÿ��>�z�@B�\CHxR                                    Bxi�G,  �          @�{������\)?�Q�AI��CE�q�������R>��@1�CIz�                                    Bxi�U�  �          @������?��HAN�HCIc�����G�>\)?�Q�CL��                                    Bxi�dx  �          @�p���
=��?�\)Ai�CF�=��
=��>���@�{CK�                                    Bxi�s  �          @����  ���?���Ag�
CE���  �   >�G�@��RCI��                                    Bxi��  �          @��R���
����?�Q�AuC@xR���
��Q�?8Q�@��CF:�                                    Bxi�j  �          @�33�����\?��Ap��CKB�����p�>k�@\)COaH                                    Bxi�  �          @����p����H?޸RA��CJ��p��#�
?z�@�CP��                                    Bxi쭶  �          @�p�������@�A��CM0������7�?J=qA
=CTff                                    Bxi�\  �          @�����\)��@
�HA�(�CL�3��\)�7�?c�
A��CT��                                    Bxi��  �          @����p��
=@�\A�ffCMxR��p��<��?z�HA'\)CU�R                                    Bxi�٨  T          @������
� ��@��A�CL�q���
�<��?�z�AG\)CV                                      Bxi��N  �          @�ff��G��ff@'
=A��CN���G��G
=?�G�AV{CW�f                                    Bxi���  �          @��R��z���@"�\AݮCL޸��z��@��?�p�AP��CVxR                                    Bxi��  �          @����Q쿁G�?�ffA�\)C?s3��Q��G�?�=qA8(�CGc�                                    Bxi�@  �          @�  ��p��O\)?��HA��HC<�f��p����?���A8��CD�
                                    Bxi�"�  �          @�
=���5?�=qA��HC;Ǯ�����?��A/\)CC                                    Bxi�1�  �          @�����þ�
=?���Au��C8�����ÿ�=q?��A2�\C?��                                    Bxi�@2  �          @�\)������R?��A��\C7k�������\?�Q�AH��C?
=                                    Bxi�N�  �          @����  ��?�=qA�(�C5Y���  �aG�?���A^�HC=�                                    Bxi�]~  �          @�����׿�(�@	��A���CBT{�����
=?��
A\��CK�f                                    Bxi�l$  �          @�p����Ϳ�@$z�A��CJaH�����5�?���Ag�CT��                                    Bxi�z�  T          @���Q���H@{A��HCH����Q��,��?��A`��CR�                                    Bxi�p  �          @�{��33�У�@��AиRCG����33�%?��AZ�RCQff                                    Bxi�  T          @����\��{@�AԸRCGk����\�&ff?�=qAb=qCQ��                                    Bxi���  �          @��R��33�Ǯ@�RA�(�CF���33�$z�?��Am�CQ0�                                    Bxi��b  �          @�ff���R�У�@p�A�\)CG!H���R� ��?�\)A>ffCP�                                    Bxi��  �          @�
=��{��{@%A�CA\)��{�\)?ٙ�A��CM��                                    Bxi�Ү  �          @�
=�������@'
=A�
=CA@ ����\)?޸RA�z�CM�                                    Bxi��T  T          @����G����@8��A�Q�CA0���G��ff?��RA�Q�CO\)                                    Bxi���  �          @����\)����@#33A�p�CA
��\)�p�?�
=A�Q�CM\                                    Bxi���  �          @�\)��Q쿞�R@5�A���CCc���Q��{?���A�ffCPǮ                                    Bxi�F  �          @�{���\��@*�HA뙚CBc����\�?޸RA�CO\                                    Bxi��  �          @�
=���H��Q�@�A�{CA�)���H��?�  AUCKL�                                    Bxi�*�  �          @�{��p���(�?�AG
=CAO\��p����>��@��CE�H                                    Bxi�98  
�          @������H��33?��Ao33C@�3���H��Q�?#�
@��HCFQ�                                    Bxi�G�  �          @�z���p���\)?��A�=qC@�\��p���z�?��A:ffCI0�                                    Bxi�V�  T          @��������@
=A��C@L�������H?��A_�
CJ�                                    Bxi�e*  �          @����  ��z�?ٙ�A��HCA��  ����?h��A=qCHG�                                    Bxi�s�  �          @�p��������?���Al(�CCY��������?�\@���CH�                                     Bxi�v  �          @�ff������\?�(�A��CBs3�����?�=qA733CJ��                                    Bxi�  �          @�\)�������\@�HA��C@������?˅A�G�CK��                                    Bxi��  �          @�����
��(�?���Al(�CJ{���
���>aG�@�
CNB�                                    Bxi�h  �          @������?�Q�AnffCJ�����!�>W
=@�CN��                                    Bxi�  T          @��
��
=�  ?�\)Aa�CLW
��
=�(Q�=�\)?+�CO��                                    Bxi�˴  �          @��\���R�˅?�p�A���CE�����R��?n{A��CM��                                    Bxi��Z  �          @�(���녿�  @�A���CA�������?��HAF�RCJ�R                                    Bxi��   �          @���(��\?�A��CDz���(��G�?fffAffCK�H                                    Bxi���  �          @�����׿޸R?�Q�A�=qCG#�������R?O\)A�CNB�                                    Bxi�L  T          @�p���Q����?�33A���CHO\��Q��#�
?8Q�@�G�CO�                                    Bxi��  T          @�ff��  ��z�?���A�  CH����  �(Q�?:�H@�z�CO��                                    Bxi�#�  �          @�=q�������?���A�33CL�����5�?   @��CR��                                    Bxi�2>  �          @�Q�����
?�G�A���CNL����8Q�>�Q�@qG�CS�H                                    Bxi�@�  �          @�����
=�{?�A�
=CM&f��
=�4z�>�ff@�\)CR��                                    Bxi�O�  �          @��H�����p�?��A��HCL������6ff?�@���CR��                                    Bxi�^0  �          @�z������p�?�Q�A�G�CL� �����8��?�@���CS
=                                    Bxi�l�  �          @��\����ff@33A���CK�H����5?8Q�@�ffCR�q                                    Bxi�{|  �          @�Q���
=�   @ ��A�G�CJ���
=�/\)?:�H@��CR�                                    Bxi�"  �          @���������?�  A�33CI���� ��?\)@��HCO{                                    Bxi��  �          @�Q���G���?�
=A�CI�H��G��(Q�?333@��CPǮ                                    Bxi�n  �          @������׿�(�?��HA�(�CJu������,(�?333@��CQk�                                    Bxi�  �          @�����{���H@
=qA�  CJ�f��{�1�?aG�AffCR��                                    Bxi�ĺ  
�          @�G����R��z�@�A��CJ)���R�0  ?k�A��CRL�                                    Bxi��`  �          @�33���\����@z�A���CI�3���\�.�R?O\)Az�CQp�                                    Bxi��  �          @����33���H@G�A���CJ��33�-p�?@  @�ffCQ5�                                    Bxi��  �          @��\���R��ff@�Aə�CH�����R�0  ?��A<��CRJ=                                    Bxi��R  �          @������Ϳ�  @p�A�33CH�������0  ?��RAO�CR�H                                    Bxi��  �          @�����(����H@\)A�ffCHB���(��.�R?��AW33CR��                                    Bxi��  �          @����Q��G�@.{A��HCIJ=��Q��8��?��HAs�
CT�                                    Bxi�+D  �          @������H����@(Q�A�CG=q���H�-p�?��HAt��CR��                                    Bxi�9�  �          @�����Q����@1�A�CG(���Q��0��?���A�
=CSu�                                    Bxi�H�  �          @�����R��(�@5�A��RCFB����R�-p�?ٙ�A�{CS:�                                    Bxi�W6  �          @������R����@:=qA�(�CF\���R�.�R?�\A�\)CSp�                                    Bxi�e�  �          @����(����
@QG�B
=CAG���(��#33@�\A��HCR5�                                    Bxi�t�  �          @�����H���@g
=B"  CB����H�1G�@#�
A��CV&f                                    Bxi��(  
�          @�=q���H���@fffB!Q�CC�\���H�3�
@!�A��
CV��                                    Bxi��  �          @��\��=q��Q�@hQ�B"  CDL���=q�7�@!G�A�ffCW&f                                    Bxi�t  �          @��H�}p���\)@q�B*33CC�=�}p��8Q�@+�A噚CX�                                    Bxi�  �          @�33�y�����
@w�B/�CB��y���6ff@3�
A�
=CX.                                    Bxi��  �          @�{���ÿL��@|(�B/ffC?B������,��@?\)A��CU��                                    Bxi��f  �          @��R���׿B�\@~�RB0��C>�������+�@A�B ��CU�3                                    Bxi��  �          @�  �z�H�333@�z�B8�C>(��z�H�.{@Mp�BCV�q                                    Bxi��  �          @�{�xQ�8Q�@�33B8�C>��xQ��-p�@J=qB  CV�R                                    Bxi��X  �          @�\)���׿Y��@}p�B/�\C?�3�����0  @>{A��RCV\)                                    Bxi��  �          @����G�����@,��A��
CI
��G��>{?���A[�
CS                                    Bxi��  �          @�33��=q��ff@1G�A�  CHn��=q�=p�?��HAg�CS�                                    Bxi�$J  �          @��
�����p�@+�Aٙ�CGn����7
=?�z�A_�CR0�                                    Bxi�2�  �          @������
�Ǯ@:�HA��
CE�����
�5�?��HA�\)CR(�                                    Bxi�A�  
�          @��������
@,��A�(�CG�����:=q?�z�A]CR��                                    Bxi�P<  �          @�(����
�?˅A{�CL�����
�4z�>.{?�\)CPٚ                                    Bxi�^�  �          @����
=�{?У�A�{CK���
=�/\)>��@!�CO��                                    Bxi�m�  �          @���������?�=qAx(�CJ{�����)��>��@!�CN�f                                    Bxi�|.  �          @�33��G��   ?�  AmCH�3��G���R>��@%�CM+�                                    Bxi��  �          @��
��\)�G�?�p�A�
=CI#���\)�'�>�ff@��
CN�H                                    Bxi�z  �          @��
���ÿ��R?�\)A��\CH������"�\>�p�@h��CM��                                    Bxi�   �          @�z������(�@�A��CK�������>{?0��@���CR�{                                    Bxi��  �          @����p���\@ffA�33CL�q��p��J�H?W
=AffCT�{                                    Bxi��l  �          @���
=���@=qA�=qCKE��
=�Dz�?xQ�A
=CS��                                    Bxi��  �          @�p������\)@#33A�ffCH�\����:=q?�(�A?33CRJ=                                    Bxi��  �          @�z����H��  @"�\A���CDu����H�&ff?�z�A]CO�                                    Bxi��^  �          @�\)�������@Dz�A��CA� ����'
=@G�A��CO�{                                    Bxi�   T          @������ÿ�z�@E�A�z�C@�q�����%�@�\A��HCO&f                                    Bxi��  �          @�Q���  ��z�@B�\A���CC����  �1G�?��A�  CQ                                    Bxi�P  �          @�33�����(�@K�A��CDn����9��?�(�A��RCR+�                                    Bxi�+�  �          @����
=����@s33B(�CA}q��
=�;�@*�HAͅCS�\                                    Bxi�:�  �          @Ϯ��z��=q@_\)BQ�CG����z��W
=@�\A�33CU8R                                    Bxi�IB  �          @�  ���
�ff@O\)A�CL�f���
�k�?���A_\)CW                                    Bxi�W�  �          @�z�����E@:=qAٙ�CU
�����z�?fffA��C]+�                                    Bxi�f�  �          @�z���
=�Vff@Q�A���CU�q��
=�~{=���?^�RCZ��                                    Bxi�u4  �          @ҏ\����S�
@A��HCT���������>�z�@"�\CZ�                                    Bxi��  �          @�  ���H�33@<��Aޣ�CLJ=���H�_\)?��AC�CVn                                    Bxi�  �          @�ff��33��
@UA�  CJ  ��33�_\)?��A�p�CVaH                                    Bxi�&  �          @�G���ff��(�@^�RBQ�CG!H��ff�QG�@A�33CUn                                    Bxi��  �          @�p������{@UB	�CGE����G
=@�A�(�CU��                                    Bxi�r  �          @����33��Q�@S33B��CH&f��33�I��?���A�ffCVp�                                    Bxi��  �          @����H��\)@^{B��CFxR���H�K�@��A��
CUQ�                                    Bxi�۾  T          @�  ��Q��Q�@<��A��CKG���Q��L(�?\Au�CW@                                     Bxi��d  �          @��\���R��G�@U�B��CG�����R�AG�@�
A���CW��                                    Bxi��
  �          @�\)��
=��33@8��A��
CL:���
=�G�?��RA}�CX}q                                    Bxi��  �          @�p����ÿG�@G
=B=qC>Y�������@��A�p�CP\                                    Bxi�V  �          @���33�n{@,(�A�ffC?u���33�(�?���A���CMxR                                    Bxi�$�  �          @�{���ÿ�z�@��A��CDz������z�?���A:�RCM�H                                    Bxi�3�  �          @�ff��33���
@�RA�CC�)��33���?���A{
=COn                                    Bxi�BH  �          @�\)�j=q�#�
@�  B<�C=�H�j=q�'�@EB
  CW�f                                    Bxi�P�  �          @���Y��=u@�G�BN33C2��Y���
�H@l��B)��CT�=                                    Bxi�_�  �          @��H�A�?O\)@��
B_(�C$���A녿ٙ�@��
BM�CQT{                                    Bxi�n:  �          @��3�
?(��@�  Be\)C&�R�3�
��\@�{BNffCTB�                                    Bxi�|�  �          @�p��333��Q�@�
=Bg�C5�
�333���@qG�B6�\C\��                                    Bxi�  �          @�
=�(�?�{@�\)Bp\)C���(���=q@�(�B}�CNB�                                    Bxi�,  �          @��H��z�?�
=@�G�B�G�C5ÿ�zῳ33@���B��3CX0�                                    Bxi��  �          @�Q�˅@�\@���B�B��˅����@�B���CUǮ                                    Bxi�x  �          @Å����@ ��@�33Bv33B�\�����#�
@��
B�ǮCK�
                                    Bxi��  �          @�(��aG�@7�@���Bm=qB�LͿaG��8Q�@��B�Q�C?��                                    Bxi���  �          @ҏ\?(�@Z=q@�G�BgB�Ǯ?(�<�@�  B�L�@<(�                                    Bxi��j  �          @�{?h��@]p�@�z�Bfz�B�aH?h��<#�
@ӅB�B�?k�                                    Bxi��  �          @�Q�?��
@Z�H@��Bd�\B��?��
��@ӅB�\C�h�                                    Bxi� �  �          @�?�G�@\��@��\Bb�B��?�G�<�@��B�L�?�=q                                    Bxi�\  �          @�?�ff@qG�@��BU�B�\?�ff>�
=@�G�B�z�A�{                                    Bxi�  �          @���?��@QG�@�=qBb�B�
=?��<��
@�  B��3?W
=                                    Bxi�,�  �          @���?��@s�
@�G�BJQ�B�ff?��?�R@�G�B��)A�G�                                    Bxi�;N  �          @�G�?�Q�@S�
@��BV�B|
=?�Q�>8Q�@���B���@�{                                    Bxi�I�  �          @�z�?��H@s�
@�z�BH��B�\?��H?0��@��B���A˙�                                    Bxi�X�  �          @�?���@�{@�(�B%��B�L�?���?�(�@�B�k�B;�                                    Bxi�g@  �          @�{?�@��H@�G�B:�HB�Q�?�?��@�{B�ǮB>�R                                    Bxi�u�  �          @�z�?Ǯ@��\@��B)
=B�.?Ǯ?��H@�ffB��=B>z�                                    Bxi�  �          @�(�?�z�@��@�=qB2��B��)?�z�?��@�ffB�z�B	�R                                    Bxi��2  �          @�z�@  @x��@�
=BE=qBo�@  ?z�@�\)B�� Ahz�                                    Bxi���  �          @�?�=q@z�H@�(�BQ��B��H?�=q>���@�33B�\)AEp�                                    Bxi��~  �          @��H?ٙ�@s33@�\)BW�B��q?ٙ�>�  @��
B��HA(�                                    Bxi��$  �          @�?�@mp�@�=qB\Q�B�aH?�=�G�@���B��@z=q                                    Bxi���  �          @�\?��H@U�@�Q�Bhz�B{p�?��H���R@ۅB���C��                                    Bxi��p  �          @ᙚ?ٙ�@\��@��Bc�RB~��?ٙ����@��HB��C�|)                                    Bxi��  �          @߮?���@Z�H@���Bf  B���?��þ8Q�@�=qB��HC�Ф                                    Bxi���  �          @��?��
@J=q@��Bi  Brff?��
��p�@���B�aHC��                                    Bxi�b  �          @���?�\)@7�@��Bsp�Brz�?�\)�&ff@�G�B�.C�)                                    Bxi�  �          @ٙ�?�{@A�@�(�Bn��Bw��?�{�   @��HB�G�C�n                                    Bxi�%�  �          @�z�?�{@O\)@��Bi33B~G�?�{���R@�{B�u�C���                                    Bxi�4T  �          @�33?��
@R�\@��Bh=qB�#�?��
�u@�p�B��fC�|)                                    Bxi�B�  �          @�p�?�
=@E�@�
=Bm{B�W
?�
=��p�@�\)B�ffC�Ф                                    Bxi�Q�  �          @��?�G�@g�@��BW=qB��
?�G�>�\)@θRB�W
A'�                                    Bxi�`F  �          @�  ?�p�@l��@�33BUp�B�\)?�p�>�{@�  B��AQ                                    Bxi�n�  �          @�ff?��@J=q@��B[=qB�  ?��>\)@�G�B���@�G�                                    Bxi�}�  �          @�
=?B�\@XQ�@��
BZ=qB��?B�\>���@�p�B�{A�G�                                    Bxi��8  �          @��?�=q@Z�H@�G�BT��B��?�=q>��@��
B���A���                                    Bxi���  �          @�  ?L��@�  @���B<�HB�L�?L��?���@��HB��
BW{                                    Bxi���  �          @���?�@���@�33B+{B�8R?�?���@���B�k�B���                                    Bxi��*  �          @�ff>�
=@�  @w�B"�B�k�>�
=?�@���B���B�
=                                    Bxi���  �          @�?
=q@�@eB33B��f?
=q@
=@�Q�B���B�\)                                    Bxi��v  �          @���?˅@���@��HB:�B�8R?˅?��@���B�\BG�                                    Bxi��  �          @�\@�@��@��B;��B|�@�?u@��B���A��R                                    Bxi���  �          @��
?�33@�\)@��RB5�
B��R?�33?���@�\)B���A��\                                    Bxi�h  �          @Ǯ?�@�{@�{B'�B��\?�?˅@�z�B�L�B@�R                                    Bxi�  �          @�ff?�z�@��@��B3{B�k�?�z�?�G�@�z�B�ffB&Q�                                    Bxi��  �          @���?�p�@��R@��RB-G�B�(�?�p�?���@�G�B�u�B+Q�                                    Bxi�-Z  �          @���?0��@��R@w
=B(�B�B�?0��?�p�@�Q�B��fB�z�                                    Bxi�<   �          @Ǯ?�\)@��@�B5�B�z�?�\)?��H@�ffB��
B%�
                                    Bxi�J�  �          @�Q�?O\)@�z�@�33B133B��?O\)?���@�Q�B��Bs\)                                    Bxi�YL  T          @ƸR?�{@w�@�G�B={B�?�{?h��@�z�B��A�{                                    Bxi�g�  �          @Ǯ?��@p��@�  BG�
B��?��?5@���B��A�=q                                    Bxi�v�             @�=q���H@��R@���B>�B�Q���H?�33@�p�B�#�B�p�                                    Bxi��>  �          @�(���G�@�(�@�33BF\)B�  ��G�?xQ�@ə�B�ffB��3                                    Bxi���  �          @ȣ׿8Q�@|(�@��BHB��8Q�?Tz�@��B��RC�                                    Bxi���  �          @ȣ׿n{@�Q�@�
=BC��B�.�n{?n{@��
B���C��                                    Bxi��0  �          @�Q���@mp�@���BU�B��q���?�@�
=B�ǮC s3                                    Bxi���  T          @�\)>��@mp�@��RBT��B��f>��?��@�p�B��B�8R                                    Bxi��|  �          @ȣ�?��@dz�@�  BI�B��3?��?
=q@�p�B�z�A�
=                                    Bxi��"  �          @��@
=q@z=q@�\)B4�Bt(�@
=q?xQ�@��
B�A�p�                                    Bxi���  �          @�Q�?���@l��@�BB
=B}?���?0��@�B��A�z�                                    Bxi��n  �          @Ǯ?�{@[�@��BL��Bvz�?�{>�p�@�{B���A5�                                    Bxi�	  �          @���?��@Z�H@�Q�BT�B�u�?��>�\)@��B���A%G�                                    Bxi��  �          @�?��
@P  @�z�Ba�
B�?��
=�\)@\B��3@g�                                    Bxi�&`  �          @�?@  @n�R@��BOB�33?@  ?�R@�33B�ǮB�                                    Bxi�5  �          @�ff?
=@l��@�BR�B���?
=?�@�z�B��)B/�                                    Bxi�C�  �          @ƸR?
=q@G
=@�=qBmffB�.?
=q���@��B��\C�K�                                    Bxi�RR  T          @��R?8Q�@Y��@��BX�B�33?8Q�>�p�@�(�B�ffAم                                    Bxi�`�  
�          @�G�?=p�@aG�@��
BV33B�\)?=p�>�G�@��B��3A���                                    Bxi�o�  �          @�=q?&ff@]p�@�{BY�B���?&ff>�Q�@���B��A�
=                                    Bxi�~D  "          @��H?Y��@`  @���BV��B��{?Y��>��@�Q�B�{A��H                                    Bxi���  T          @�33?^�R@l(�@���BN=qB�p�?^�R?!G�@�  B���BG�                                    Bxi���  �          @��R?
=@Z=q@�33BZ  B�8R?
=>�Q�@�p�B���A��                                    Bxi��6  �          @�\)?�\@R�\@�ffB`ffB�B�?�\>L��@�{B�ffA�(�                                    Bxi���  
�          @��
?(�@p  @���BN(�B���?(�?.{@���B���BA\)                                    Bxi�ǂ  �          @�{?�G�@w
=@�\)B:
=B�  ?�G�?p��@��HB�B�A���                                    Bxi��(  �          @ə�?���@���@�33B0G�B���?���?�@��\B�p�A���                                    Bxi���  "          @���@(�@�  @z�HB�RBp��@(�?Ǯ@�=qBz�Bff                                    Bxi��t  
�          @�\)@ ��@��\@mp�BQ�Bop�@ ��?޸R@�Br��B
�                                    Bxi�  �          @�@=q@��@mp�BBs(�@=q?�(�@�p�BuffB(�                                    Bxi��  "          @�z�@ff@���@dz�Bp�Bw�\@ff?�\)@�33Br�RB(�                                    Bxi�f  T          @��@33@��\@Z�HBB}�@33@
=@��Bn33B)�                                    Bxi�.  T          @ƸR?�p�@�=q@z=qB�B���?�p�?У�@��HB��Bp�                                    Bxi�<�  �          @��H?��@��@|��B!ffB�
=?��?��@��HB��
B�                                    Bxi�KX  "          @�  ?��@�33@~{B&ffB�?��?�z�@���B�8RBz�                                    Bxi�Y�  T          @�\)?��@z�H@�(�B/  B�
=?��?�
=@�=qB���B�                                    Bxi�h�  
�          @��?��@�=q@|��B%�B�.?��?��@�  B�B�\                                    Bxi�wJ  �          @�
=@   @�z�@r�\B�B�z�@   ?��@���B���BQ�                                    Bxi���  �          @�=q@\)@�z�@c�
B=qB{��@\)?��@��\Bu�B�H                                    Bxi���  T          @��@
�H@���@eBG�B|=q@
�H?�G�@�G�Bx�RB                                      Bxi��<  �          @�@��@��\@X��B��B|ff@��?�z�@��BrG�B$
=                                    Bxi���  �          @���?���@p  @��\B3�HB��)?���?��@�{B��BQ�                                    Bxi���  �          @�\)?�33@L(�@���BD(�Bl?�33>��@��B�� A`z�                                    Bxi��.  �          @���?�G�@dz�@�Q�B>�
B�B�?�G�?O\)@�  B�u�A�=q                                    Bxi���  "          @�G�?z�H@�z�@X��BQ�B�W
?z�H?�  @���B��qBr�                                    Bxi��z  T          @�p�>\@���@0��A���B���>\@ ��@���Bw��B��q                                    Bxi��   �          @�{=�Q�@�\)@Z�HB�RB�u�=�Q�?�@��
B���B�{                                    Bxi�	�  "          @��R>�(�@�\)@Z�HB(�B�aH>�(�?���@�(�B�  B�L�                                    Bxi�l  "          @�\)?#�
@|��@o\)B,33B�� ?#�
?�z�@�  B�#�B���                                    Bxi�'  
�          @�Q�?�\)@p  @w�B2�RB��?�\)?�@�  B��qB8�
                                    Bxi�5�  
�          @���?#�
@s�
@|��B6�\B�
=?#�
?�
=@�33B��Bv��                                    Bxi�D^  "          @�Q�?(�@qG�@}p�B8=qB��?(�?��@�33B��
Bw�
                                    Bxi�S  
�          @�=q>Ǯ@X��@��
BO��B�>Ǯ?(�@�  B�Bc��                                    Bxi�a�  
�          @�(�?�@\(�@���BN��B�� ?�?!G�@���B��BH
=                                    Bxi�pP  
�          @��
?333@q�@��\B:�B��?333?��@�ffB�\)BeQ�                                    Bxi�~�  �          @��\?\(�@n{@���B:�B�?\(�?��@���B��BK��                                    Bxi���  T          @��\?^�R@z=q@x��B0B��?^�R?�ff@��B��B`��                                    Bxi��B  T          @�
=?�@�ff@QG�Bp�B��H?�?�\)@�\)B�(�Bh{                                    Bxi���  "          @��
���@�@�(�B�\B��H����Q�@�
=B�\Ci��                                    Bxi���  �          @����z�@!G�@�  Be�B�\)��z�W
=@��B�33C<O\                                    Bxi��4  
�          @��
�Tz�@%�@�33Bm�B׀ �Tz�L��@���B��CA�\                                    Bxi���  �          @�Q��@Q�@�  B�33B��녿(��@��B�ǮCe:�                                    